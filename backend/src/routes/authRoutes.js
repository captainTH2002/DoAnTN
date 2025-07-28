const express = require('express')
const router = express.Router()
const authController = require('../controllers/authController')
const authMiddleware = require('../middleware/auth')

router.post('/auth/register', authController.register)
router.post('/auth/login', authController.login)
router.get('/users/me', authMiddleware, authController.getMe)

module.exports = router
