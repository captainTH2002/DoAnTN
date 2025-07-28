const express = require('express')
const router = express.Router()

router.use(require('./authRoutes'))
router.use(require('./fileRoutes'))
router.use(require('./userRoutes'))
router.use(require('./chatRoutes'))
router.use(require('./conversationRoutes'))
router.use(require('./messageRoutes'))
router.use(require('./adminRoutes'))

module.exports = router
