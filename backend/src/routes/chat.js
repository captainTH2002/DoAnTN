const express = require('express');
const router = express.Router();
const File = require('../models/File');
const { createEmbedding } = require('../services/embedding');
const { askGemini } = require('../services/gemini');

router.post('/', async (req, res) => {
  const { question } = req.body;
  const queryEmbedding = await createEmbedding(question);

  // Atlas Vector Search
  const results = await File.aggregate([
    {
      $vectorSearch: {
        queryVector: queryEmbedding,
        path: "embedding",
        numCandidates: 100,
        limit: 5,
        index: "myVectorIndex"
      }
    }
  ]);

  const context = results.map(d => d.text).join('\n');
  const answer = await askGemini(context, question);
  res.json({ answer });
});

module.exports = router;
