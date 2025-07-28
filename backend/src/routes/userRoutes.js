const express = require('express')
const router = express.Router()
const userController = require('../controllers/userController')
const authMiddleware = require('../middleware/auth')
const checkRole = require('../middleware/checkRole')

router.get('/users', authMiddleware, checkRole('admin'), userController.getUsers)
router.put('/users/:id', authMiddleware, checkRole('admin'), userController.updateUser)
router.delete('/users/:id', authMiddleware, checkRole('admin'), userController.deleteUser)

module.exports = router
