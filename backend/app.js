const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const fileRoutes = require('./src/routes/fileRoutes');
require('dotenv').config(); // đọc file .env

const app = express();
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static('uploads'));
app.use('/api', fileRoutes);

// Kết nối MongoDB Atlas
mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log('✅ Connected to MongoDB Atlas'))
.catch(err => console.error('❌ MongoDB connection error:', err));

module.exports = app;
