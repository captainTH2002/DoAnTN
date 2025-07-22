const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fileController = require('../controllers/fileController');
const authController = require('../controllers/authController')
const authMiddleware = require('../middleware/auth')
const checkRole = require('../middleware/checkRole')
const { searchTopK } = require('../services/vectorService'); // 👈 import thêm
const chatRoutes = require("./chatRoutes.js");
const userController = require('../controllers/userController');

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    // Khôi phục lại UTF-8 nếu tên gốc bị lỗi encoding
    const decodedName = Buffer.from(file.originalname, 'latin1').toString('utf8');
    cb(null, Date.now() + '-' + decodedName);
  },
});

const upload = multer({ storage });

router.get('/file/:id', fileController.downloadFile);

router.get('/file', fileController.getFiles);
router.post('/file', upload.single('file'), fileController.uploadFile);
router.delete('/file/:id', authMiddleware, checkRole(['admin', 'editor']), fileController.deleteFile);
router.put('/file/:id', authMiddleware, checkRole(['admin', 'editor']), fileController.updateFile);
router.use("/chat", chatRoutes);

router.get('/file/search', async (req, res) => {
  try {
    const q = req.query.q;
    const results = await searchTopK(q);
    res.json(results);
  } catch (err) {
    console.error(err);
    res.status(500).send(err.message);
  }
});

router.post('/auth/register', authController.register)
router.post('/auth/login', authController.login)
router.get('/users/me', authMiddleware, authController.getMe)


router.get('/conversations', async (req, res) => {
  const userId = req.user._id  // nếu dùng JWT / middleware xác thực
  const conversations = await Conversation.find({ user: userId }).sort({ createdAt: -1 })
  res.json(conversations)
})

// Tạo conversation
router.post('/conversations', async (req, res) => {
  const userId = req.user._id
  const newConv = await Conversation.create({ user: userId })
  res.json(newConv)
})

router.get('/messages/:conversationId', async (req, res) => {
  const messages = await Message.find({ conversation: req.params.conversationId }).sort({ createdAt: 1 })
  res.json(messages)
})

router.post('/chat', async (req, res) => {
  const { conversationId, question } = req.body
  const userMsg = await Message.create({ conversation: conversationId, role: 'user', message: question })

  // 👉 Đây bạn gọi RAG hoặc ChatModel để sinh câu trả lời
  const answer = "Đây là câu trả lời giả lập"

  const botMsg = await Message.create({ conversation: conversationId, role: 'bot', message: answer })

  res.json({ answer })
})

// Lưu message
router.post('/messages', async (req, res) => {
  const { conversationId, role, message } = req.body
  const msg = await Message.create({ conversation: conversationId, role, message })
  res.json(msg)
})


// Admin-only
router.get('/admin/secret', authMiddleware, checkRole('admin'), (req, res) => {
  res.json({ message: 'Chỉ admin mới xem được!' })
})
// USERS (admin only)
router.get('/users', userController.getUsers)
router.put('/users/:id', authMiddleware, checkRole('admin'), userController.updateUser)
router.delete('/users/:id', authMiddleware, checkRole('admin'), userController.deleteUser)


router.get('/stats', fileController.getFileStats);
router.get('/recent', fileController.getRecentFiles);
module.exports = router;
