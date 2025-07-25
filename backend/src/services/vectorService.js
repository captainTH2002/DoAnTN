const axios = require('axios');
const pool = require("../pg.js");
const { getEmbedding } = require("../utils/embedding.js");
const fs = require('fs');
const fsp = require('fs').promises; // fs.promises
const path = require('path');
const mammoth = require('mammoth');
const pdf = require('pdf-parse');
const FormData = require('form-data');
const { execFile } = require('child_process');

function ocrImageWithPython(imagePath) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.resolve(__dirname, '../../text_detect_ocr/img2text.py');
    execFile('python', [scriptPath, imagePath], (error, stdout, stderr) => {
      if (error) {
        console.error('Python OCR error:', stderr || error);
        return reject(stderr || error);
      }

      // Sau khi Python chạy xong, đọc file kết quả
      const outputPath = path.join(path.dirname(imagePath), 'ocr_result.txt');
      fs.readFile(outputPath, 'utf8', (err, data) => {
        if (err) {
          console.error("Read OCR result error:", err);
          return reject(err);
        }
        resolve(data.trim());
      });
    });
  });
}


async function extractText(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (['.jpg', '.jpeg', '.png'].includes(ext)) {
    // Gọi Python OCR
    return await ocrImageWithPython(filePath);
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
