export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const backendUrl = process.env.AZURE_BACKEND_URL;
    

    if (!backendUrl) {
      
      return res.status(500).json({ error: 'Backend URL not configured' });
    }

    // Get the raw body as buffer
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const body = Buffer.concat(chunks);

    
    
    

    const response = await fetch(`${backendUrl}/detect-food`, {
      method: 'POST',
      body: body,
      headers: {
        'Content-Type': req.headers['content-type'],
        'Content-Length': req.headers['content-length'],
      },
      duplex: 'half',
    });

    

    const data = await response.json();
    res.status(response.status).json(data);
  } catch (error) {
    console.error('Detect food error:', error);
    res.status(500).json({ error: 'Internal server error', details: error.message });
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
};
