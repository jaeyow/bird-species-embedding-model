import { Button, Box, Card, CardMedia, Tooltip } from '@mui/material';
import { useRef, useState } from 'react';
import axios from 'axios';

// Test function to verify CORS
const testCors = async () => {
  try {
    const response = await fetch('http://localhost:8000/test-cors', {
      method: 'GET',
      mode: 'cors',
      headers: {
        'Accept': 'application/json',
      },
    });
    const data = await response.json();
    console.log('CORS test response:', data);
  } catch (error) {
    console.error('CORS test failed:', error);
  }
};

export function ImageUpload({ setResults, setLoading }) {
  const fileInputRef = useRef();
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Create preview URL for the uploaded image
    setPreviewUrl(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      console.log('Making request to API...');
      
      // Test CORS first
      await testCors();
      
      const response = await axios.post(
        'http://localhost:8000/get_similar_birds',
        formData,
        {
          params: {
            limit: 8,
            metric: 'cosine'
          },
          headers: {
            'Content-Type': 'multipart/form-data',
            'Accept': 'application/json',
          },
          // Disable credentials for testing
          withCredentials: false,
        }
      );
      console.log('Response received:', response);
      setResults(response.data);
    } catch (error) {
      console.error('Full error object:', error);
      if (error.response) {
        console.error('Response data:', error.response.data);
        console.error('Response status:', error.response.status);
        console.error('Response headers:', error.response.headers);
      } else if (error.request) {
        console.error('Request was made but no response received:', error.request);
      } else {
        console.error('Error setting up request:', error.message);
      }
      alert('Error uploading image. Check console for details.');
    } finally {
      setLoading(false);
    }
  };

  // Cleanup preview URL when component unmounts
  const cleanup = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
  };

  return (
    <Box sx={{ textAlign: 'center', mb: 2 }}>
      <input
        type="file"
        accept="image/*"
        hidden
        ref={fileInputRef}
        onChange={handleUpload}
      />
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
        <Tooltip title="Upload a bird image to search similar birds" arrow>
          <Button
            variant="contained"
            onClick={() => fileInputRef.current.click()}
            size="large"
          >
            Upload Bird Image
          </Button>
        </Tooltip>
        
        {previewUrl && (
          <Card sx={{ maxWidth: 300, width: '100%', mt: 2 }}>
            <Box sx={{ 
              position: 'relative',
              height: 0,
              paddingTop: '100%', // 1:1 Aspect ratio
              bgcolor: '#f5f5f5', // Light grey background
            }}>
              <CardMedia
                component="img"
                image={previewUrl}
                alt="Uploaded bird"
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
          </Card>
        )}
      </Box>
    </Box>
  );
} 