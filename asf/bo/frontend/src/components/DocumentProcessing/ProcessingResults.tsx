import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Tabs,
  Tab,
  Divider,
  Button,
  Chip,
  List,
  ListItem,
  ListItemText,
  Card,
  CardContent,
  Grid
} from '@mui/material';
import {
  Description as DocumentIcon,
  LocalOffer as TagIcon,
  Link as LinkIcon,
  Download as DownloadIcon,
  Share as ShareIcon
} from '@mui/icons-material';
import { ProcessingResult } from '../../hooks/useDocumentProcessing';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`processing-tabpanel-${index}`}
      aria-labelledby={`processing-tab-${index}`}
      {...other}
      style={{ height: '100%', overflow: 'auto' }}
    >
      {value === index && (
        <Box sx={{ p: 2, height: '100%' }}>
          {children}
        </Box>
      )}
    </div>
  );
}

interface ProcessingResultsProps {
  result: ProcessingResult | null;
  onExport: (format: string) => void;
}

/**
 * Component for displaying document processing results
 */
const ProcessingResults: React.FC<ProcessingResultsProps> = ({
  result,
  onExport
}) => {
  const [tabValue, setTabValue] = useState(0);

  if (!result) {
    return (
      <Paper sx={{ p: 3, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography variant="body1" color="textSecondary">
          No results to display. Process a document to see results here.
        </Typography>
      </Paper>
    );
  }

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const { results } = result;
  const entities = results.entities || [];
  const relations = results.relations || [];
  const sections = results.sections || [];
  const references = results.references || [];

  return (
    <Paper sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="h6">Processing Results</Typography>
          <Box>
            <Button
              startIcon={<DownloadIcon />}
              size="small"
              onClick={() => onExport('json')}
              sx={{ mr: 1 }}
            >
              Export JSON
            </Button>
            <Button
              startIcon={<DownloadIcon />}
              size="small"
              onClick={() => onExport('pdf')}
            >
              Export PDF
            </Button>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1 }}>
          <Chip
            icon={<DocumentIcon />}
            label={`Processing time: ${result.processing_time.toFixed(2)}s`}
            size="small"
          />
          <Chip
            icon={<TagIcon />}
            label={`Entities: ${entities.length}`}
            size="small"
            color="primary"
          />
          <Chip
            icon={<LinkIcon />}
            label={`Relations: ${relations.length}`}
            size="small"
            color="secondary"
          />
        </Box>
      </Box>

      <Divider />

      <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="Content" />
          <Tab label="Entities" />
          <Tab label="Relations" />
          <Tab label="Sections" />
          <Tab label="References" />
          {results.summary && <Tab label="Summary" />}
        </Tabs>

        <Box sx={{ flex: 1, overflow: 'hidden' }}>
          {/* Content Tab */}
          <TabPanel value={tabValue} index={0}>
            <Box sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '0.875rem' }}>
              {results.content || 'No content available'}
            </Box>
          </TabPanel>

          {/* Entities Tab */}
          <TabPanel value={tabValue} index={1}>
            {entities.length === 0 ? (
              <Typography variant="body2" color="textSecondary">
                No entities found in the document.
              </Typography>
            ) : (
              <List>
                {entities.map((entity) => (
                  <ListItem key={entity.id} divider>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Typography variant="body1" component="span">
                            {entity.text}
                          </Typography>
                          <Chip
                            label={entity.label}
                            size="small"
                            color="primary"
                            sx={{ ml: 1 }}
                          />
                          {entity.confidence !== undefined && (
                            <Chip
                              label={`Confidence: ${(entity.confidence * 100).toFixed(1)}%`}
                              size="small"
                              sx={{ ml: 1 }}
                            />
                          )}
                        </Box>
                      }
                      secondary={`Position: ${entity.start}-${entity.end}`}
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </TabPanel>

          {/* Relations Tab */}
          <TabPanel value={tabValue} index={2}>
            {relations.length === 0 ? (
              <Typography variant="body2" color="textSecondary">
                No relations found in the document.
              </Typography>
            ) : (
              <List>
                {relations.map((relation) => (
                  <ListItem key={relation.id} divider>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Typography variant="body1" component="span">
                            {relation.head}
                          </Typography>
                          <Chip
                            label={relation.relation}
                            size="small"
                            color="secondary"
                            sx={{ mx: 1 }}
                          />
                          <Typography variant="body1" component="span">
                            {relation.tail}
                          </Typography>
                        </Box>
                      }
                      secondary={
                        relation.confidence !== undefined
                          ? `Confidence: ${(relation.confidence * 100).toFixed(1)}%`
                          : undefined
                      }
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </TabPanel>

          {/* Sections Tab */}
          <TabPanel value={tabValue} index={3}>
            {sections.length === 0 ? (
              <Typography variant="body2" color="textSecondary">
                No sections found in the document.
              </Typography>
            ) : (
              <List>
                {sections.map((section) => (
                  <Card key={section.id} sx={{ mb: 2 }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        {section.title || 'Untitled Section'}
                      </Typography>
                      <Typography variant="body2" color="textSecondary" gutterBottom>
                        Position: {section.start}-{section.end}
                      </Typography>
                      <Divider sx={{ my: 1 }} />
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                        {section.content}
                      </Typography>
                    </CardContent>
                  </Card>
                ))}
              </List>
            )}
          </TabPanel>

          {/* References Tab */}
          <TabPanel value={tabValue} index={4}>
            {references.length === 0 ? (
              <Typography variant="body2" color="textSecondary">
                No references found in the document.
              </Typography>
            ) : (
              <List>
                {references.map((reference) => (
                  <ListItem key={reference.id} divider>
                    <ListItemText
                      primary={reference.text}
                      secondary={
                        <Box sx={{ mt: 1 }}>
                          {reference.doi && (
                            <Chip
                              icon={<LinkIcon />}
                              label={`DOI: ${reference.doi}`}
                              size="small"
                              sx={{ mr: 1 }}
                              clickable
                              component="a"
                              href={`https://doi.org/${reference.doi}`}
                              target="_blank"
                            />
                          )}
                          {reference.url && (
                            <Chip
                              icon={<LinkIcon />}
                              label="URL"
                              size="small"
                              clickable
                              component="a"
                              href={reference.url}
                              target="_blank"
                            />
                          )}
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </TabPanel>

          {/* Summary Tab */}
          {results.summary && (
            <TabPanel value={tabValue} index={5}>
              <Grid container spacing={3}>
                {results.summary.abstract && (
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Abstract
                        </Typography>
                        <Typography variant="body2">
                          {results.summary.abstract}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                )}

                {results.summary.key_findings && (
                  <Grid item xs={12} md={6}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Key Findings
                        </Typography>
                        <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                          {results.summary.key_findings}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                )}

                {results.summary.conclusion && (
                  <Grid item xs={12} md={6}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Conclusion
                        </Typography>
                        <Typography variant="body2">
                          {results.summary.conclusion}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                )}
              </Grid>
            </TabPanel>
          )}
        </Box>
      </Box>
    </Paper>
  );
};

export default ProcessingResults;
