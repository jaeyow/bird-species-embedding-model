import { 
  Grid, 
  Card, 
  CardMedia, 
  CardContent, 
  Typography, 
  Box, 
  CircularProgress,
  Modal,
  Paper,
  IconButton
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useState } from 'react';

export function ResultsGrid({ results, loading }) {
  const [selectedBird, setSelectedBird] = useState(null);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (results.length === 0) {
    return null;
  }

  const handleCardClick = (bird) => {
    setSelectedBird(bird);
  };

  const handleCloseModal = () => {
    setSelectedBird(null);
  };

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h5" component="h2" gutterBottom>
        Similar Birds
      </Typography>
      <Grid container spacing={3}>
        {results.map((bird, index) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
            <Card
              onClick={() => handleCardClick(bird)}
              sx={{
                height: 400,
                display: 'flex',
                flexDirection: 'column',
                transition: 'transform 0.2s ease-in-out',
                '&:hover': {
                  transform: 'scale(1.05)',
                  cursor: 'pointer',
                  boxShadow: 3
                }
              }}
            >
              <Box sx={{ 
                position: 'relative',
                height: '75%',
                bgcolor: '#f5f5f5',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden'
              }}>
                <CardMedia
                  component="img"
                  image={`http://localhost:8000/birds/${bird.image_path}`}
                  alt={bird.label}
                  sx={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                  }}
                />
              </Box>
              <CardContent sx={{ 
                flexGrow: 1,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center'
              }}>
                <Typography variant="h8" component="div" noWrap>
                  {bird.label}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Similarity: {(bird.similarity * 100).toFixed(2)}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Modal */}
      <Modal
        open={selectedBird !== null}
        onClose={handleCloseModal}
        aria-labelledby="bird-modal-title"
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Paper
          sx={{
            position: 'relative',
            maxWidth: '80%',
            maxHeight: '90vh',
            width: 800,
            bgcolor: 'background.paper',
            boxShadow: 24,
            p: 4,
            outline: 'none',
            overflow: 'auto'
          }}
        >
          {selectedBird && (
            <>
              <IconButton
                onClick={handleCloseModal}
                sx={{
                  position: 'absolute',
                  right: 8,
                  top: 8,
                }}
              >
                <CloseIcon />
              </IconButton>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                <Typography variant="h4" component="h2" id="bird-modal-title">
                  {selectedBird.label}
                </Typography>
                
                <Box sx={{ 
                  width: '100%',
                  position: 'relative',
                  borderRadius: 1,
                  overflow: 'hidden',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <CardMedia
                    component="img"
                    image={`http://localhost:8000/birds/${selectedBird.image_path}`}
                    alt={selectedBird.label}
                    sx={{
                      width: '70%',
                      height: '70%',
                      objectFit: 'cover',
                    }}
                  />
                </Box>

                <Box>
                  <Typography variant="h6" gutterBottom>
                    Similarity Score: {(selectedBird.similarity * 100).toFixed(2)}%
                  </Typography>
                  {/* <Typography variant="body1">
                    This {selectedBird.label.toLowerCase()} shares significant visual characteristics 
                    with your uploaded image, including similar patterns, colors, and features 
                    typical of this species.
                  </Typography> */}
                </Box>
              </Box>
            </>
          )}
        </Paper>
      </Modal>
    </Box>
  );
} 