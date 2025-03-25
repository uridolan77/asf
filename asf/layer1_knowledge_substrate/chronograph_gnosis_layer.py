import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import Lorentz, Manifold, PoincaréBall
from geoopt.manifolds.stereographic import Stereographic
from pydantic import BaseModel, Field, validator
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric_temporal import TemporalConv

# Database Integrations (using async drivers from previous example)
import asyncpg
from neo4j import AsyncGraphDatabase

# Constants (from previous example, adjust as needed)
RETRY_BACKOFF_SECONDS = 1
MAX_RETRIES = 3
KAFKA_TOPIC = "chrono-ingest"  # Assuming this topic is also used for embeddings
CACHE_TTL_SECONDS = 60
TOKEN_ALGORITHM = "RS256"  # Assuming same security setup

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("chronograph-gnosis-v2")


# --- Configuration (using Pydantic, similar to previous example) ---
class Neo4jConfig(BaseModel):
    uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    user: str = Field("neo4j", env="NEO4J_USER")
    password: str = Field("password", env="NEO4J_PASSWORD")


class TimescaleConfig(BaseModel):
    dbname: str = Field("chronograph", env="TIMESCALE_DBNAME")
    user: str = Field("tsadmin", env="TIMESCALE_USER")
    password: str = Field("secret", env="TIMESCALE_PASSWORD")
    host: str = Field("localhost", env="TIMESCALE_HOST")
    port: int = Field(5432, env="TIMESCALE_PORT")


class EmbeddingConfig(BaseModel):
    euclidean_dim: int = 256
    hyperbolic_dim: int = 128
    curvature: float = 0.7  # Example curvature value


class ReasoningConfig(BaseModel):
    min_support: float = 0.1  # Example min support for rule mining
    max_rule_length: int = 5
    attention_heads: int = 8


class GnosisConfig(BaseModel):
    neo4j: Neo4jConfig = Neo4jConfig()
    timescale: TimescaleConfig = TimescaleConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    reasoning: ReasoningConfig = ReasoningConfig()
    tcn_channels: List[int] = [32, 32, 64]
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2
    sage_hidden_channels: int = 128
    sage_out_channels: int = 256
    temporal_edge_channels: int = 32
    learning_rate: float = 1e-4
    batch_size: int = 32


# --- Exceptions (from previous example) ---
class ChronoError(Exception):
    """Base Chronograph exception."""

    pass
class ChronoDatabaseError(ChronoError):
    """Database interaction error."""

    pass
class ChronoGnosisError(ChronoError):
      """Gnosis Layer Error"""

# --- Database Manager (from previous example, adapted for async) ---
class DatabaseManager:
    def __init__(self, config: GnosisConfig):
        self.timescale_config = config.timescale
        self.neo4j_config = config.neo4j
        self.timescale_pool = None
        self.neo4j_driver = None

    async def connect(self):
        """Establish database connections."""
        self.timescale_pool = await asyncpg.create_pool(
            database=self.timescale_config.dbname,
            user=self.timescale_config.user,
            password=self.timescale_config.password,
            host=self.timescale_config.host,
            port=self.timescale_config.port,
            min_size=1,
            max_size=10,
        )
        self.neo4j_driver = AsyncGraphDatabase.driver(
            self.neo4j_config.uri,
            auth=(self.neo4j_config.user, self.neo4j_config.password),
        )
        await self.neo4j_driver.verify_connectivity()
        logger.info("Database connections established.")

    async def close(self):
        """Close database connections."""
        if self.timescale_pool:
            await self.timescale_pool.close()
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        logger.info("Database connections closed.")

    async def fetch_temporal_data(self, entity_id: str) -> Dict:
        """Fetches time-series data for an entity from TimescaleDB."""
        query = """
            SELECT
                EXTRACT(EPOCH FROM (timestamp - LAG(timestamp, 1, timestamp) OVER (ORDER BY timestamp))) AS delta,
                data->>'value' AS value
            FROM entities
            WHERE id = $1
            ORDER BY timestamp;
        """
        try:
            async with self.timescale_pool.acquire() as conn:
                results = await conn.fetch(query, entity_id)
            if not results:
                return {"deltas": [], "values": []}

            deltas = [
                float(row["delta"]) if row["delta"] is not None else 0.0
                for row in results
            ]
            values = [float(row["value"]) for row in results]

            return {"deltas": deltas, "values": values}

        except asyncpg.PostgresError as e:
            logger.error(f"Error fetching temporal data for entity {entity_id}: {e}")
            raise ChronoDatabaseError(f"Failed to fetch temporal data: {e}")

    async def fetch_subgraph(self, entity_id: str, hops: int = 2) -> List[Dict]:
        """Fetches a multi-hop subgraph from Neo4j for a given entity ID."""

        query = f"""
        MATCH (n:Entity {{id: $entity_id}})
        CALL apoc.path.expandConfig(n, {{
            maxLevel: {hops},
            uniqueness: 'NODE_GLOBAL'
        }})
        YIELD path
        WITH n as source_node, nodes(path)[1..] as target_nodes, relationships(path) as rels
        RETURN source_node, target_nodes, rels
        """

        try:
            result = await self.neo4j_driver.execute_query(
                query, {"entity_id": entity_id}
            )
            # Neo4j results are now a list of dictionaries
            return result  # no processing done
        except Exception as e:
            logger.error(f"Error fetching subgraph for entity {entity_id}: {e}")
            raise ChronoDatabaseError(f"Failed to fetch subgraph: {e}")

    async def get_all_entity_ids(self) -> List[str]:
            """Retrieves all entity IDs from the Neo4j database."""
            query = "MATCH (e:Entity) RETURN e.id as id"
            try:
                result = await self.neo4j_driver.execute_query(query)
                return [record['id'] for record in result]
            except Exception as e:
                logger.error(f"Error fetching all entity IDs: {e}")
                raise ChronoDatabaseError("Failed to fetch entity IDs")

# --- Neural Network Components ---

class MultiHeadTemporalAttention(nn.Module):
    """Multi-Head Attention with temporal encoding."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # Simple linear layer for temporal encoding, consider more sophisticated methods
        self.temporal_encoding = nn.Linear(1, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        time_deltas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, embed_dim)
        # time_deltas shape: (batch_size, sequence_length)

        if time_deltas is not None:
          # Add temporal encoding to the input
          time_deltas = time_deltas.unsqueeze(-1)  # (batch, seq_len, 1)
          temporal_embeddings = self.temporal_encoding(time_deltas) #simple
          x = x + temporal_embeddings

        attn_output, _ = self.attention(x, x, x)  # Self-attention
        attn_output = self.dropout(attn_output)
        x = self.layer_norm(x + attn_output)  # Residual connection + LayerNorm
        return x


class TemporalEncoder(nn.Module):
    """Temporal pattern encoder using dilated TCNs and attention."""

    def __init__(self, config: GnosisConfig, database_manager: DatabaseManager):
        super().__init__()
        self.database = database_manager
        self.tcn_channels = config.tcn_channels
        self.kernel_size = config.tcn_kernel_size
        self.dropout_rate = config.tcn_dropout
        self.final_embedding_size = config.embedding.euclidean_dim + config.embedding.hyperbolic_dim


        # Dynamically create TCN layers with increasing dilation
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # Start with 1 input channel (the value)
        for i, out_channels in enumerate(self.tcn_channels):
            dilation = 2**i
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    self.kernel_size,
                    padding="same",  # Use 'same' padding to keep output length
                    dilation=dilation,
                )
            )
            in_channels = out_channels  # Update in_channels for the next layer

        self.attention = MultiHeadTemporalAttention(
            embed_dim=in_channels, num_heads=4, dropout=0.1
        )
        # Project to a consistent embedding size
        self.projection = nn.Linear(in_channels, self.final_embedding_size)


    async def encode(self, entity_id: str) -> torch.Tensor:
        """Encode temporal evolution of an entity."""
        data = await self.database.fetch_temporal_data(entity_id)

        # Convert to tensor
        if not data["values"]:  # Handle empty data
             return torch.zeros(1,self.final_embedding_size, device='cuda' if torch.cuda.is_available() else 'cpu' ) # Return zero tensor if no data

        values = torch.tensor(data["values"], dtype=torch.float32)
        time_deltas = torch.tensor(data["deltas"], dtype=torch.float32)


        # Temporal convolution
        x = values.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        for conv in self.conv_layers:
            x = conv(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)


        # Temporal attention  (x now has shape [batch, channels, seq_len])
        x = x.permute(0, 2, 1) #  [batch, seq_len, channels]
        x = self.attention(x, time_deltas)
        # Global Max Pooling to get a single vector representation
        x = torch.max(x, dim=1)[0] # Max pool over the sequence length
        #Project
        x = self.projection(x)
        return x


class TemporalEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
      super().__init__()
      #placeholder, no weights to update
      self.in_channels = in_channels
      self.out_channels = out_channels

    def forward(self, x, edge_index, edge_time):
      return x

class GraphEncoder(nn.Module):
    """Structural encoder using GraphSAGE with temporal edges."""

    def __init__(self, config: GnosisConfig, database_manager: DatabaseManager):
        super().__init__()
        self.database = database_manager
        self.hidden_channels = config.sage_hidden_channels
        self.out_channels = config.sage_out_channels
        self.temporal_edge_channels = config.temporal_edge_channels
        self.final_embedding_size = config.embedding.euclidean_dim + config.embedding.hyperbolic_dim

        self.conv1 = SAGEConv(-1, self.hidden_channels)  # -1 for automatic inference
        self.conv2 = SAGEConv(self.hidden_channels, self.out_channels)
        self.temporal_conv = TemporalEdgeConv(
            self.out_channels, self.temporal_edge_channels
        )
        self.projection = nn.Linear(self.out_channels, self.final_embedding_size)


    async def _neo4j_to_pyg(self, subgraph: List[Dict]) -> Data:
        """Converts a Neo4j subgraph to a PyG Data object."""

        # Create a mapping from node IDs to indices
        node_ids = set()
        for record in subgraph:
            node_ids.add(record["source_node"]["id"])
            for target_node in record["target_nodes"]:
                node_ids.add(target_node["id"])
        node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

        # Create edge index
        edge_index_source = []
        edge_index_target = []
        for record in subgraph:
            source_index = node_id_to_index[record["source_node"]["id"]]
            for target_node in record["target_nodes"]:
                target_index = node_id_to_index[target_node["id"]]
                edge_index_source.append(source_index)
                edge_index_target.append(target_index)

        edge_index = torch.tensor(
            [edge_index_source, edge_index_target], dtype=torch.long
        )

        # Create feature matrix (using one-hot encoding for simplicity)
        num_nodes = len(node_id_to_index)
        x = torch.eye(num_nodes, dtype=torch.float)

        # Determine the index of the center node (assuming the first node is the center)
        center_node_index = node_id_to_index[subgraph[0]["source_node"]["id"]]
        mask = torch.tensor([center_node_index], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, mask=mask)

    async def encode(self, entity_id: str) -> torch.Tensor:
        """Encode structural context of an entity."""
        subgraph = await self.database.fetch_subgraph(entity_id)
        if not subgraph:
            return torch.zeros(1, self.final_embedding_size, device='cuda' if torch.cuda.is_available() else 'cpu')  # Handle empty subgraph
        data = await self._neo4j_to_pyg(subgraph)
        # Move data to the appropriate device
        data = data.to('cuda' if torch.cuda.is_available() else 'cpu')

        x = self.conv1(data.x, data.edge_index)
        x = F.gelu(x)
        x = self.conv2(x, data.edge_index)
        x = self.temporal_conv(
            x, data.edge_index, None
        )  # Assuming you have edge times
        x = x[data.mask]  # Return center entity embedding

        x = self.projection(x)
        return x

class CrossManifoldAttention(nn.Module):
    def __init__(self, euclidean_dim, hyperbolic_dim):
      super().__init__()
      self.euclidean_dim = euclidean_dim
      self.hyperbolic_dim = hyperbolic_dim
      self.W_euclidean = nn.Linear(euclidean_dim, euclidean_dim)
      self.W_hyperbolic = nn.Linear(hyperbolic_dim, hyperbolic_dim)
      self.attn_weights = nn.Linear(euclidean_dim+hyperbolic_dim,1)

    def forward(self, euclidean_emb, hyperbolic_emb):

      q_e = self.W_euclidean(euclidean_emb)
      k_e = self.W_euclidean(euclidean_emb)
      v_e = euclidean_emb

      q_h = self.W_hyperbolic(hyperbolic_emb)
      k_h = self.W_hyperbolic(hyperbolic_emb)
      v_h = hyperbolic_emb

      # Concatenate for attention calculation
      attn_input_e = torch.cat((q_e, k_e), dim=-1)
      attn_input_h = torch.cat((q_h, k_h), dim=-1)
      # Attention weights
      attn_weights_e = torch.sigmoid(self.attn_weights(attn_input_e))
      attn_weights_h = torch.sigmoid(self.attn_weights(attn_input_h))

      # Weighted sum
      fused_emb = attn_weights_e * v_e + attn_weights_h * v_h
      return fused_emb

class HybridSpaceTransformer(nn.Module):
    """Manifold learning across Euclidean and hyperbolic spaces."""

    def __init__(self, config: GnosisConfig):
        super().__init__()
        self.euclidean_dim = config.embedding.euclidean_dim
        self.hyperbolic_dim = config.embedding.hyperbolic_dim
        self.curvature = config.embedding.curvature

        # Euclidean projection
        self.proj_euclidean = nn.Sequential(
            nn.Linear(
                config.embedding.euclidean_dim + config.embedding.hyperbolic_dim, 512
            ),  # Input size adjusted
            nn.GELU(),
            nn.Linear(512, self.euclidean_dim),
        )

        # Hyperbolic projection (Poincaré ball)
        self.manifold = PoincaréBall(c=self.curvature)
        self.proj_hyperbolic = nn.Sequential(
            nn.Linear(
                config.embedding.euclidean_dim + config.embedding.hyperbolic_dim, 512
            ),  # Input size adjusted
            nn.GELU(),
            nn.Linear(512, self.hyperbolic_dim),
        )

        # Attention-based fusion
        self.fusion = CrossManifoldAttention(
            euclidean_dim=self.euclidean_dim, hyperbolic_dim=self.hyperbolic_dim
        )

    def forward(
        self, temporal_emb: torch.Tensor, graph_emb: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Transform to hybrid embedding space."""
        # Concatenate features
        combined = torch.cat([temporal_emb, graph_emb], dim=-1)

        # Project to both spaces
        euclidean = self.proj_euclidean(combined)
        hyperbolic = self.manifold.expmap0(
            self.proj_hyperbolic(combined)
        )  # Ensure hyperbolic is on manifold

        # Fuse representations
        fused = self.fusion(euclidean, hyperbolic)

        return {
            "euclidean": euclidean,
            "hyperbolic": hyperbolic,
            "fused": fused,
        }

class TemporalReasoner(nn.Module): #Stubbed out for now, filled in later
    """Placeholder for Temporal Reasoning."""
    def __init__(self, config: GnosisConfig):
      super().__init__()
      self.config = config
      pass
    def forward(self, x:Dict)->Dict:
      return x
# --- Data Management ---

class TemporalGraphDataset(Dataset):
    """Dataset for loading temporal graph data."""

    def __init__(
        self,
        database_manager: DatabaseManager,
        entity_ids: Optional[List[str]] = None,
    ):
        super().__init__()
        self.database = database_manager
        self.entity_ids = entity_ids

        # If entity_ids are not provided, fetch them
        if self.entity_ids is None:
          self.entity_ids = asyncio.run(self.database.get_all_entity_ids())


    def __len__(self):
        if self.entity_ids:
          return len(self.entity_ids)
        return 0


    def __getitem__(self, idx):

      if not self.entity_ids:
        return {}

      entity_id = self.entity_ids[idx]

      # Fetch data (using async methods, but we need to run them synchronously in __getitem__)
      loop = asyncio.get_event_loop() #get current event loop
      temporal_data = loop.run_until_complete(self.database.fetch_temporal_data(entity_id))
      graph_data = loop.run_until_complete(self.database.fetch_subgraph(entity_id))
      pyg_graph = loop.run_until_complete(self.database._neo4j_to_pyg(graph_data))


      # Convert to tensors
      if temporal_data and temporal_data["values"]:  # Handle cases with no temporal data
        temporal_values = torch.tensor(temporal_data["values"], dtype=torch.float).unsqueeze(-1) #match shape
        temporal_deltas = torch.tensor(temporal_data["deltas"], dtype=torch.float)
      else:
        temporal_values = torch.tensor([], dtype=torch.float)
        temporal_deltas = torch.tensor([], dtype=torch.float)

      # Return as a dictionary (similar to previous examples)
      return {
          "entity_id": entity_id,
          "temporal_values": temporal_values,
          "temporal_deltas": temporal_deltas,
          "graph": pyg_graph, #full pyg_graph
          # Placeholder targets.  Replace with your actual target data.
          "targets": {
              "euclidean": torch.randn(256),  # Example target
              "hyperbolic": torch.randn(128),  # Example target
          },
      }

# --- Training Infrastructure ---
class HybridManifoldLoss(nn.Module):
    """Loss function spanning Euclidean and hyperbolic spaces."""

    def __init__(self):
        super().__init__()
        self.euclidean_loss = nn.CosineEmbeddingLoss()
        self.hyperbolic_loss = nn.MSELoss() #placeholder

    def forward(self, pred, target):
      #Ensure target is not empty
      if target['euclidean'].numel() == 0 or target['hyperbolic'].numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device = pred['euclidean'].device)

      euclidean_loss = self.euclidean_loss(
          pred["euclidean"],
          target["euclidean"],
          torch.ones(pred["euclidean"].size(0), device=pred["euclidean"].device),
      )


      hyperbolic_loss = self.hyperbolic_loss(
          pred["hyperbolic"], target["hyperbolic"]
      ) # Replace with a hyperbolic distance loss

      return 0.7 * euclidean_loss + 0.3 * hyperbolic_loss  # Weighted sum


class GnosisTrainer(pl.LightningModule):
    """PyTorch Lightning training module."""

    def __init__(self, config: GnosisConfig, database_manager: DatabaseManager):
        super().__init__()
        self.config = config
        self.temporal_encoder = TemporalEncoder(config, database_manager)
        self.graph_encoder = GraphEncoder(config, database_manager)
        self.hybrid_transformer = HybridSpaceTransformer(config)
        self.reasoner = TemporalReasoner(config)  # Placeholder
        self.loss_fn = HybridManifoldLoss()
        self.database = database_manager #for the

    def forward(self, batch) -> Dict:
        """Forward pass through the entire model."""

        # 1. Encode Temporal Data

        # Pad sequences to the maximum length within the batch
        temporal_values_padded = torch.nn.utils.rnn.pad_sequence([data['temporal_values'] for data in batch], batch_first=True, padding_value=0)
        temporal_deltas_padded = torch.nn.utils.rnn.pad_sequence([data['temporal_deltas'] for data in batch], batch_first=True, padding_value=0)

        # Check if sequences are empty, and create zero tensors if so.
        if temporal_values_padded.numel() == 0:
            temporal_embeddings = torch.zeros((len(batch), self.temporal_encoder.final_embedding_size), device=self.device) #correct
        else:
            # Pass the padded sequences to the encoder
            temporal_embeddings = self.temporal_encoder.attention(
                self.temporal_encoder.projection(
                torch.max(
                torch.relu(
                    torch.stack([
                    conv(temporal_values_padded.unsqueeze(1)) for conv in self.temporal_encoder.conv_layers
                    ], dim=0)).squeeze(0)
                ),
                 dim = 1)[0] #reduce
            ,temporal_deltas_padded)

        # 2. Encode Graph Data
        graph_embeddings = torch.stack([
            self.graph_encoder(data['graph'].x, data['graph'].edge_index, data['graph'].edge_attr if hasattr(data['graph'], 'edge_attr') else None)[data['graph'].mask]
            if data['graph'].num_nodes > 0 else  # Check if graph is empty
            torch.zeros((1,self.graph_encoder.final_embedding_size), device=self.device) #one set of embeddings
            for data in batch
        ])


        # 3. Hybrid Space Transformation
        hybrid_embeddings = self.hybrid_transformer(
            temporal_embeddings, graph_embeddings.squeeze(1)
        )

        # 4. Reasoning (Placeholder)
        # reasoning_output = self.reasoner(hybrid_embeddings)

        return hybrid_embeddings # Return hybrid embeddings

    def training_step(self, batch, batch_idx):
        outputs = self(batch)  # Use the forward method
        targets = {
            'euclidean': torch.stack([b['targets']['euclidean'] for b in batch]).to(self.device),
            'hyperbolic': torch.stack([b['targets']['hyperbolic'] for b in batch]).to(self.device)
        }
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        targets = {
            'euclidean': torch.stack([b['targets']['euclidean'] for b in batch]).to(self.device),
            'hyperbolic': torch.stack([b['targets']['hyperbolic'] for b in batch]).to(self.device)
        }        
        loss = self.loss_fn(outputs, targets)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
# --- Main ChronoGnosis Layer ---

class ChronoGnosisLayer:
    """Temporal Knowledge Graph Embedding System."""

    def __init__(self, config: GnosisConfig):
        self.config = config
        self.database = DatabaseManager(self.config)
        self.model = GnosisTrainer(self.config, self.database)  # Initialize with config

    async def startup(self):
        """Initialize connections and load model."""
        await self.database.connect()
        # Load model (if available)

    async def shutdown(self):
        """Gracefully shut down services."""
        await self.database.close()

    async def generate_embeddings(self, entity_ids: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Generate hybrid embeddings for entities."""

        self.model.eval()  # Set model to evaluation mode
        embeddings = {}

        with torch.no_grad():  # Disable gradient calculations
            for entity_id in entity_ids:
                 # Create a single-item dataset for the entity
                dataset = TemporalGraphDataset(self.database, entity_ids=[entity_id])

                if len(dataset) > 0:
                  dataloader = DataLoader(dataset, batch_size=1)
                  # Get the batch from the dataloader (there will only be one batch)
                  batch = next(iter(dataloader))

                  # Move batch to the correct device
                  if isinstance(batch, list):
                    batch = [
                        {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                        for k,v in b.items()}
                        for b in batch
                    ]
                  else:
                    batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

                  # Generate embeddings
                  entity_embeddings = self.model(batch)
                  # Store embeddings (converting tensors to lists for easier handling)
                  embeddings[entity_id] = {
                      k: v.tolist() for k, v in entity_embeddings.items()
                  }
                else:
                  embeddings[entity_id] = {} #Handle empty embeddings

        return embeddings

# --- Main Execution ---
async def main():
    """Main function to run the Gnosis layer."""
    config = GnosisConfig()
    gnosis_layer = ChronoGnosisLayer(config)

    try:
        await gnosis_layer.startup()

        # --- Example: Generate Embeddings ---
        entity_ids = ["entity_1", "entity_2"]  # Replace with your entity IDs
        embeddings = await gnosis_layer.generate_embeddings(entity_ids)
        print("Embeddings:", embeddings)

        # --- Example: Train Model ---
        all_entity_ids = await gnosis_layer.database.get_all_entity_ids()
        dataset = TemporalGraphDataset(gnosis_layer.database, entity_ids=all_entity_ids)

        # Split the data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0) #num_workers to zero for debugging
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

        #Training
        trainer = pl.Trainer(
            max_epochs=10,  # Adjust as needed
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
        )
        trainer.fit(gnosis_layer.model, train_loader, val_loader)
        print("Training Complete!")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        await gnosis_layer.shutdown()

if __name__ == "__main__":
    asyncio.run(main())