const express = require("express");
const router = express.Router();
const chatController = require("../controllers/chatController");
const authMiddleware = require("../middleware/auth");

// Danh sách conversation của user
router.get("/conversations", authMiddleware, chatController.getConversations);

// Tạo conversation mới
router.post("/conversations", authMiddleware, chatController.createConversation);

// Lấy tin nhắn của 1 cuộc hội thoại
router.get("/messages/:conversationId", authMiddleware, chatController.getMessages);

// Gửi câu hỏi, lưu message, gọi Gemini
router.post("/chat/:conversationId", authMiddleware, chatController.askQuestion);

// Nếu muốn thêm API test không cần token
router.post("/chat", chatController.askQuick);

module.exports = router