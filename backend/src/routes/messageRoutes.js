const express = require('express')
const router = express.Router()
const { Message } = require('../models/Conversation')

// Lấy message
router.get('/messages/:conversationId', async (req, res) => {
  const messages = await Message.find({ conversation: req.params.conversationId }).sort({ createdAt: 1 })
  res.json(messages)
})

// Lưu message
router.post('/messages', async (req, res) => {
  const { conversationId, role, message } = req.body
  const msg = await Message.create({ conversation: conversationId, role, message })
  res.json(msg)
})

module.exports = router
