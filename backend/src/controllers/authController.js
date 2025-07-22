const bcrypt = require('bcrypt')
const jwt = require('jsonwebtoken')
const User = require('../models/User')

const JWT_SECRET = process.env.JWT_SECRET || 'your_jwt_secret'

// Đăng ký
exports.register = async (req, res) => {
  try {
    const { username, email, password } = req.body

    // Check tồn tại
    const existingUser = await User.findOne({ email })
    if (existingUser) {
      return res.status(400).json({ message: 'Email đã được đăng ký' })
    }

    // Hash password
    const salt = await bcrypt.genSalt(10)
    const passwordHash = await bcrypt.hash(password, salt)

    // Lưu
    const newUser = await User.create({
      username,
      email,
      passwordHash
    })

    res.status(201).json({ message: 'Đăng ký thành công' })
  } catch (err) {
    console.error(err)
    res.status(500).json({ message: 'Server error' })
  }
}

// Đăng nhập
exports.login = async (req, res) => {
  try {
    const { email, password } = req.body
    const user = await User.findOne({ email })
    if (!user) {
      return res.status(401).json({ message: 'Email hoặc mật khẩu không đúng' })
    }

    const isMatch = await bcrypt.compare(password, user.passwordHash)
    if (!isMatch) {
      return res.status(401).json({ message: 'Email hoặc mật khẩu không đúng' })
    }
    if (user.status !== 'active') return res.status(403).json({ message: 'Tài khoản bị khóa hoặc cấm' })

    // Tạo token
    const token = jwt.sign(
      { id: user._id, role: user.role },
      JWT_SECRET,
      { expiresIn: '1h' }
    )

    res.json({
      token,
      user: {
        _id: user._id,
        username: user.username,
        email: user.email,
        role: user.role
      }
    })
  } catch (err) {
    console.error(err)
    res.status(500).json({ message: 'Server error' })
  }
}

// Lấy thông tin user đang đăng nhập
exports.getMe = async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select('-passwordHash')
    res.json(user)
  } catch (err) {
    console.error(err)
    res.status(500).json({ message: 'Server error' })
  }
}
