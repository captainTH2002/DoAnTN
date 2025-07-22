const { GoogleGenerativeAI } = require("@google/generative-ai");
require('dotenv').config();
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
console.log("Gemini API KEY:", process.env.GEMINI_API_KEY);

async function askGemini(question, contexts=[]) {
  const prompt = `
Bạn là trợ lý AI. Dưới đây là thông tin tham khảo:
${contexts.map((c, i) => `(${i+1}) ${c}`).join("\n")}

Câu hỏi: ${question}
Vui lòng trả lời dựa trên thông tin trên.
  `;

  const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
  const result = await model.generateContent(prompt);
  return result.response.text();
}

module.exports = { askGemini };
