import torch
import torch.nn as nn

class TemporalPatternDetector:
    """
    Detects patterns in temporal sequences using recurrent neural networks.
    """
    def __init__(self, input_size, hidden_size=64):
        # GRU-based recurrent model for temporal pattern detection
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.pattern_classifier = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
    
    def forward(self, sequence_tensor):
        """Process temporal sequence to detect patterns"""
        # sequence_tensor shape: [batch_size, seq_length, input_size]
        output, hidden = self.rnn(sequence_tensor)
        # Use last hidden state for pattern classification
        pattern_score = torch.sigmoid(self.pattern_classifier(hidden.squeeze(0)))
        return pattern_score, hidden
    
    def detect_patterns(self, sequence_data):
        """
        Analyze sequence data to detect significant temporal patterns
        Returns pattern score and pattern type
        """
        # Convert sequence to tensor
        sequence_tensor = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pattern_score, _ = self.forward(sequence_tensor)
        return pattern_score.item()
