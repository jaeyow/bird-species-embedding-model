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
            <Card>
              <Box sx={{ 
                position: 'relative',
                height: 0,
                paddingTop: '100%', // 1:1 Aspect ratio
                bgcolor: '#f5f5f5', // Light grey background
              }}>
                <CardMedia
                  component="img"
                  image={`http://localhost:8000/birds/${bird.image_path}`}
                  alt={bird.label}
                  sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                  }}
                />
              </Box>
              <CardContent>
                <Typography variant="h8" component="div">
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