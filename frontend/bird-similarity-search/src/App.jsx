import { Container, Typography, Box } from '@mui/material';
import { ImageUpload } from './components/ImageUpload';
import { ResultsGrid } from './components/ResultsGrid';
import { useState } from 'react';

function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  return (
    <Container maxWidth="lg">
      <Box 
        sx={{ 
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          minHeight: '100vh',
          pt: 8, // Add more top padding
          px: 2,  // Add horizontal padding
          maxWidth: '800px',  // Set a max width for better layout
          margin: '0 auto'    // Center the box horizontally
        }}
      >
        <Typography 
          variant="h3" 
          component="h1" 
          gutterBottom 
          sx={{ 
            mb: 4,
            fontWeight: 500,
            textAlign: 'center'
          }}
        >
          Bird Similarity Search
        </Typography>
        <ImageUpload setResults={setResults} setLoading={setLoading} />
        <ResultsGrid results={results} loading={loading} />
      </Box>
    </Container>
  );
}

export default App; 