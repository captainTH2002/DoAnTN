const jwt = require('jsonwebtoken')
const JWT_SECRET = process.env.JWT_SECRET || 'your_jwt_secret'

function authMiddleware(req, res, next) {
  const authHeader = req.headers['authorization']
  const token = authHeader && authHeader.split(' ')[1]
  if (!token) return res.status(401).json({ message: 'Không có token' })
  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ message: 'Token không hợp lệ' })
    req.user = user
    next()
  })
}

module.exports = authMiddleware
