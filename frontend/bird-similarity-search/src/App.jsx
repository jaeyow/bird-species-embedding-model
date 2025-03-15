import { Container, Typography, Box, AppBar, Toolbar } from '@mui/material';
import { ImageUpload } from './components/ImageUpload';
import { ResultsGrid } from './components/ResultsGrid';
import { useState } from 'react';

function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  return (
    <>
      <AppBar position="static" color="primary" elevation={0}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Bird Similarity Search
          </Typography>
        </Toolbar>
      </AppBar>
      <Container 
        maxWidth={false}
        disableGutters
      >
        <Box 
          sx={{ 
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            minHeight: 'calc(100vh - 64px)',
            pt: 6,
            px: 4,
            width: '45%',
            margin: '0 auto'
          }}
        >
          <ImageUpload setResults={setResults} setLoading={setLoading} />
          <ResultsGrid results={results} loading={loading} />
        </Box>
      </Container>
    </>
  );
}

export default App; 