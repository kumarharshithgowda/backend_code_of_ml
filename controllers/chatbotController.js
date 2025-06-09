import { spawn } from 'child_process';

export const handleChatbotQuery = async (req, res) => {
  console.log("Received chatbot query:", req.body);
  const userInput = req.body.message;
  
  if (!userInput || userInput.trim() === '') {
    console.log("Empty message received");
    return res.status(400).json({ 
      error: 'Please provide a message to the chatbot.' 
    });
  }

  try {
    console.log(`Spawning Python process with input: "${userInput}"`);
    const python = spawn('python3', ['backend/ml/predictor.py', userInput]);
    
    let responseData = '';
    let errorData = '';

    python.stdout.on('data', (data) => {
      const chunk = data.toString();
      console.log(`Python stdout: ${chunk}`);
      responseData += chunk;
    });

    python.stderr.on('data', (data) => {
      const chunk = data.toString();
      errorData += chunk;
      console.error(`Python stderr: ${chunk}`);
    });

    python.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
      console.log(`Response data: ${responseData.substring(0, 100)}...`);
      
      if (code !== 0) {
        console.error(`Python process failed with code ${code}`);
        return res.status(500).json({ 
          error: 'An error occurred while processing your request.',
          details: errorData || 'No error details available'
        });
      }
      
      // Format the response for better display in the frontend
      const formattedResponse = responseData.replace(/\n/g, '<br>');
      console.log(`Sending response to client: ${formattedResponse.substring(0, 100)}...`);
      res.json({ response: formattedResponse });
    });

    // Handle timeout
    setTimeout(() => {
      python.kill();
      res.status(504).json({ 
        error: 'Request timed out. The stock data might be taking too long to fetch.' 
      });
    }, 30000); // 30 seconds timeout
  } catch (error) {
    console.error('Error spawning Python process:', error);
    res.status(500).json({ 
      error: 'Failed to process your request. Please try again later.' 
    });
  }
};
