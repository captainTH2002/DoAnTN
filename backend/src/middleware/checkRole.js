function checkRole(roles) {
  return (req, res, next) => {
    const allowedRoles = Array.isArray(roles) ? roles : [roles]
    if (!allowedRoles.includes(req.user.role)) {
      return res.status(403).json({ message: 'Bạn không có quyền thực hiện thao tác này' })
    }
    next()
  }
}
module.exports = checkRole
