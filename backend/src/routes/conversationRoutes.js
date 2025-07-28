const express = require('express')
const router = express.Router()
const authMiddleware = require('../middleware/auth')
const { Conversation } = require('../models/Conversation')

router.get('/conversations', authMiddleware, async (req, res) => {
  const userId = req.user._id
  const conversations = await Conversation.find({ user: userId }).sort({ createdAt: -1 })
  res.json(conversations)
})

router.post('/conversations', authMiddleware, async (req, res) => {
  const userId = req.user._id
  const newConv = await Conversation.create({ user: userId })
  res.json(newConv)
})

module.exports = router
