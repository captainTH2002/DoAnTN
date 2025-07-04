const axios = require('axios');
const pool = require("../pg.js");
const { getEmbedding } = require("../utils/embedding.js");
const fs = require('fs');
const fsp = require('fs').promises; // fs.promises
const path = require('path');
const mammoth = require('mammoth');
const pdf = require('pdf-parse');
const FormData = require('form-data');

async function extractText(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (['.jpg', '.jpeg', '.png'].includes(ext)) {
    // Gửi file ảnh sang API OCR
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    const res = await axios.post('http://127.0.0.1:8002/ocr', form, {
      headers: form.getHeaders()
    });
    return res.data.text;
  }
  if (ext === '.txt') {
    return await fsp.readFile(filePath, 'utf8');
  }
  else if (ext === '.docx') {
    const result = await mammoth.extractRawText({ path: filePath });
    return result.value;
  }
  else if (ext === '.pdf') {
    let dataBuffer = await fsp.readFile(filePath);
    let data = await pdf(dataBuffer);
    return data.text;
  }
  else {
    throw new Error("Unsupported file type: " + ext);
  }
}

async function processFileContent(filePath) {
  const text = await extractText(filePath);   // tự đọc file theo loại
  const embedding = await getEmbedding(text); // gọi API embedding
  const vectorLiteral = `[${embedding.join(",")}]`; // cho pgvector

  const result = await pool.query(
    "INSERT INTO documents (content, embedding) VALUES ($1, $2) RETURNING id",
    [text, vectorLiteral]
  );
  const pgId = result.rows[0].id;
  console.log("DEBUG: pgId =", pgId);
  console.log("typeof pgId:", typeof pgId);
  return { text, pgId };
}

async function searchTopK(textQuery) {
  const embedding = await getEmbedding(textQuery);
  const vectorLiteral = `[${embedding.join(",")}]`;

  const result = await pool.query(
    "SELECT id, content FROM documents ORDER BY embedding <-> $1 LIMIT 5",
    [vectorLiteral]
  );
  return result.rows;
}

module.exports = {
  processFileContent,
  searchTopK
};
