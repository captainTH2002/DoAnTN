const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fileController = require('../controllers/fileController');
const { searchTopK } = require('../services/vectorService'); // 👈 import thêm

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/'); // upload vào thư mục uploads/
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});
const upload = multer({ storage });

router.get('/file', fileController.getFiles);
router.post('/file', upload.single('file'), fileController.uploadFile);
router.delete('/file/:id', fileController.deleteFile);
router.put('/file/:id', fileController.updateFile);

router.get('/file/search', async (req, res) => {
  try {
    const q = req.query.q;
    const results = await searchTopK(q);
    res.json(results);
  } catch (err) {
    console.error(err);
    res.status(500).send(err.message);
  }
});


module.exports = router;
