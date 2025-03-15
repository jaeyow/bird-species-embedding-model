import { Container, Typography, Box, AppBar, Toolbar, ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { ImageUpload } from './components/ImageUpload';
import { ResultsGrid } from './components/ResultsGrid';
import { useState } from 'react';

function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  // Create a light theme
  const theme = createTheme({
    palette: {
      mode: 'light',
      background: {
        default: '#ffffff',
        paper: '#ffffff'
      }
    }
  });

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline /> {/* This will apply the base styles */}
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
            width: '67%',
            margin: '0 auto',
            bgcolor: 'background.default'
          }}
        >
          <ImageUpload setResults={setResults} setLoading={setLoading} />
          <ResultsGrid results={results} loading={loading} />
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App; 