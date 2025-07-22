const User = require('../models/User')

// GET /users?page=1&limit=10
exports.getUsers = async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1
    const limit = parseInt(req.query.limit) || 10
    const skip = (page - 1) * limit

    const total = await User.countDocuments()
    const users = await User.find()
      .skip(skip)
      .limit(limit)
      .sort({ createdAt: -1 })
      .select('-passwordHash') // không trả về hash

    res.json({
      total,
      page,
      limit,
      totalPages: Math.ceil(total / limit),
      data: users
    })
  } catch (err) {
    console.error(err)
    res.status(500).send(err.message)
  }
}

// PUT /users/:id
exports.updateUser = async (req, res) => {
  try {
    const user = await User.findById(req.params.id)
    if (!user) return res.status(404).json({ message: 'Không tìm thấy user.' })

    const { username, role, status } = req.body
    if (username) user.username = username
    if (role) user.role = role
    if (status) user.status = status

    await user.save()
    res.json(user)
  } catch (err) {
    console.error(err)
    res.status(500).send(err.message)
  }
}

// DELETE /users/:id
exports.deleteUser = async (req, res) => {
  try {
    const user = await User.findById(req.params.id)
    if (!user) return res.status(404).json({ message: 'Không tìm thấy user.' })

    await User.deleteOne({ _id: req.params.id })
    res.json({ message: 'Đã xóa user thành công.' })
  } catch (err) {
    console.error(err)
    res.status(500).send(err.message)
  }
}
