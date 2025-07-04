const axios = require("axios");

async function getEmbedding(text) {
  const res = await axios.post("http://127.0.0.1:8001/embed", { text });
  const embedding = res.data.embedding.map(Number);

  return embedding;
}

module.exports = { getEmbedding };
