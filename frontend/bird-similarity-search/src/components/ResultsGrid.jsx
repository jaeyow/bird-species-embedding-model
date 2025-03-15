import { Grid, Card, CardMedia, CardContent, Typography, Box, CircularProgress } from '@mui/material';

export function ResultsGrid({ results, loading }) {
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

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h5" component="h2" gutterBottom>
        Similar Birds
      </Typography>
      <Grid container spacing={3}>
        {results.map((bird, index) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
            <Card
              sx={{
                height: 400, // Fixed height for all cards
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
                height: '75%', // Fixed proportion for image container
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
    </Box>
  );
} 