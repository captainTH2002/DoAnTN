const axios = require('axios');
async function askGemini(context, question) {
  const prompt = `Context:\n${context}\n\nQuestion:\n${question}`;
  const resp = await axios.post(
    'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
    { contents: [{ parts: [{ text: prompt }] }] },
    { params: { key: process.env.GEMINI_API_KEY }}
  );
  return resp.data.candidates[0].content.parts[0].text;
}
module.exports = { askGemini };
