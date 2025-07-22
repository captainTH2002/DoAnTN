const mongoose = require('mongoose')

const userSchema = new mongoose.Schema({
  username: { type: String, required: true, trim: true },
  email: { type: String, required: true, unique: true, lowercase: true },
  passwordHash: { type: String, required: true },
  role: { type: String, enum: ['user', 'admin'], default: 'user' },
  status: { type: String, enum: ['active', 'inactive', 'banned'], default: 'active' },
  createdAt: { type: Date, default: Date.now }
})

module.exports = mongoose.model('User', userSchema)
