const express = require('express')
const router = express.Router()
const authMiddleware = require('../middleware/auth')
const checkRole = require('../middleware/checkRole')

router.get('/admin/secret', authMiddleware, checkRole('admin'), (req, res) => {
  res.json({ message: 'Chỉ admin mới xem được!' })
})

module.exports = router
