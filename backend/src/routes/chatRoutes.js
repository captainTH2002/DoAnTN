const express = require("express");
const { askGemini } = require("../services/askGemini.js");
const { searchTopK } = require('../services/vectorService');

const router = express.Router();

router.post("/", async (req, res) => {
  const { question } = req.body;
  console.log("==== question ====");
  console.log(question);
  // Giả sử bạn đã có vectorSearch
  // const contexts = await vectorSearch(question);
  const contexts = await searchTopK(question); 
  console.log("==== Contexts ====");
  console.log(contexts);

  const contextTexts = contexts.map(c => c.content);
  console.log("==== contextTexts ====");
  console.log(contextTexts);
  try {
    const answer = await askGemini(question, contextTexts);
    res.json({ answer, contexts: question }); // trả về để client thấy
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Có lỗi xảy ra khi gọi Gemini" });
  }
});

module.exports = router;
