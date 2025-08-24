export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const backendUrl = process.env.AZURE_BACKEND_URL;
    
    if (!backendUrl) {
      return res.status(500).json({ error: 'Backend URL not configured' });
    }

    const response = await fetch(`${backendUrl}/get-portion-recommendation`, {
      method: 'POST',
      body: JSON.stringify(req.body),
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();
    res.status(response.status).json(data);
  } catch (error) {
    console.error('Portion recommendation error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}
