const mongoose = require('mongoose');

const fileSchema = new mongoose.Schema({
  name: String,             // Tên file gốc (VD: "hopdong.jpg")
  size: Number,             // Dung lượng (byte)
  type: String,             // jpg, png, pdf...
  path: String,             // Đường dẫn trong server (VD: "uploads/123-hopdong.jpg")

  createdAt: { type: Date, default: Date.now },
  documentType: {
    type: String,
    default: 'Chưa xác định'
  },

  extractedText: String,    // Text OCR được từ ảnh (sau này bạn dùng mô hình trích xuất thông tin)
  embeddingId: Number,      // ID trong Vector DB (pinecone_id, milvus_pk...)

  tags: [String],           // VD: ["hợp đồng", "2025"]
  uploadedBy: {             // Khi phân quyền user
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User'
  }
});

module.exports = mongoose.model('File', fileSchema);
