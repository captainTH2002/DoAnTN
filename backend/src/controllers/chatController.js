const Conversation = require("../models/Conversation");
const Message = require("../models/Message");
const { askGemini } = require("../services/askGemini");
const { searchTopK } = require("../services/vectorService");

// Gọi nhanh không cần lưu
exports.askQuick = async (req, res) => {
  const { question } = req.body;
  if (!question || typeof question !== "string" || question.trim() === "") {
    return res.status(400).json({ error: "Câu hỏi không hợp lệ." });
  }

  try {
    const contexts = await searchTopK(question);
    const contextTexts = contexts.map(c => c.content);
    const answer = await askGemini(question, contextTexts);
    res.json({ answer, contexts });
  } catch (err) {
    console.error("❌ Lỗi gọi Gemini:", err);
    res.status(500).json({ error: "Đã xảy ra lỗi khi xử lý câu hỏi." });
  }
};

// Tạo cuộc trò chuyện mới
exports.createConversation = async (req, res) => {
  try {
    const conversation = new Conversation({ user: req.user.id });
    await conversation.save();
    res.status(201).json(conversation);
  } catch (err) {
    res.status(500).json({ error: "Không thể tạo cuộc trò chuyện mới" });
  }
};

// Lấy danh sách conversation của user
exports.getConversations = async (req, res) => {
  try {
    const conversations = await Conversation.find({ user: req.user.id }).sort({ createdAt: -1 });
    res.json(conversations);
  } catch (err) {
    res.status(500).json({ error: "Lỗi khi lấy danh sách cuộc trò chuyện" });
  }
};

// Lấy tin nhắn của 1 cuộc hội thoại
exports.getMessages = async (req, res) => {
  try {
    const messages = await Message.find({ conversation: req.params.conversationId }).sort({ createdAt: 1 });
    res.json(messages);
  } catch (err) {
    res.status(500).json({ error: "Lỗi khi lấy tin nhắn" });
  }
};

// Gửi câu hỏi trong 1 conversation (và lưu user + bot message)
exports.askQuestion = async (req, res) => {
  const { conversationId } = req.params;
  const { question } = req.body;

  if (!question || typeof question !== "string" || question.trim() === "") {
    return res.status(400).json({ error: "Câu hỏi không hợp lệ." });
  }

  try {
    const contexts = await searchTopK(question);
    const contextTexts = contexts.map(c => c.content);
    const answer = await askGemini(question, contextTexts);

    // Lưu câu hỏi và câu trả lời
    await Message.create({ conversation: conversationId, role: "user", content: question });
    await Message.create({ conversation: conversationId, role: "bot", content: answer });
    
    res.json({ answer });
  } catch (err) {
    console.error("❌ Lỗi khi gửi câu hỏi:", err);
    res.status(500).json({ error: "Đã xảy ra lỗi khi xử lý câu hỏi." });
  }
};
