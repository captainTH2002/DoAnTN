const File = require("../models/File");
const path = require("path");
const pool = require("../pg.js");
const fs = require('fs')
const mime = require('mime-types')
const { format, getISOWeek, getYear } = require('date-fns')
const normalizeType = (type) => {
  if (!type) return 'unknown'
  const t = type.toLowerCase()
  if (['jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif', 'image/jpeg','image/png'].includes(t)) return 'image'
  if (['pdf', 'docx'].includes(t)) return t
  return 'other'
}


exports.downloadFile = async (req, res) => {
  try {
    const file = await File.findById(req.params.id);
    if (!file) return res.status(404).send("Không tìm thấy file trong database.");

    // Tạo đường dẫn tuyệt đối, cross-platform
    const absolutePath = path.resolve(file.path.replace(/\\/g, '/'));

    console.log("📁 Absolute path:", absolutePath);

    if (!fs.existsSync(absolutePath)) {
      console.log("🚫 File không tồn tại:", absolutePath);
      return res.status(404).send("File không tồn tại trên ổ đĩa.");
    }

    // Set type đúng
    const mimeType = mime.lookup(absolutePath) || "application/octet-stream";
    res.setHeader("Content-Type", mimeType);

    // Gửi file về client
    res.download(absolutePath, file.name, (err) => {
      if (err) {
        console.error("❌ Lỗi gửi file:", err);
        res.status(500).send("Không thể tải file.");
      }
    });
  } catch (err) {
    console.error("🔥 Error:", err);
    res.status(500).send(err.message);
  }
};


exports.getFiles = async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1
    const limit = parseInt(req.query.limit) || 10
    const skip = (page - 1) * limit

    const {
      name,
      type,
      typeList,
      sizeMin,
      sizeMax,
      dateFrom,
      dateTo,
    } = req.query

    const query = {}

    // Tìm theo tên tài liệu
    if (name) {
      query.name = { $regex: name, $options: 'i' }
    }

    // Tìm theo 1 loại file cụ thể (ví dụ: "pdf")
    if (type) {
      query.type = type
    }

    // Tìm theo nhiều loại (ví dụ: ảnh: jpg,jpeg,png)
    if (typeList) {
      try {
        const extensions = typeList.split(',').map((ext) => ext.trim())
        query.type = { $in: extensions }
      } catch (e) {
        console.warn('typeList parse lỗi:', typeList)
      }
    }

    // Tìm theo kích thước
    if (sizeMin || sizeMax) {
      query.size = {}
      if (sizeMin) query.size.$gte = parseFloat(sizeMin) * 1024
      if (sizeMax) query.size.$lte = parseFloat(sizeMax) * 1024
    }

    // Tìm theo khoảng thời gian
    if (dateFrom || dateTo) {
      query.createdAt = {}
      if (dateFrom) query.createdAt.$gte = new Date(dateFrom)
      if (dateTo) {
        const toDate = new Date(dateTo)
        toDate.setHours(23, 59, 59, 999) // lấy hết ngày
        query.createdAt.$lte = toDate
      }
    }

    // Tổng số kết quả
    const total = await File.countDocuments(query)

    // Kết quả phân trang
    const files = await File.find(query)
      .skip(skip)
      .limit(limit)
      .sort({ createdAt: -1 })

    res.json({
      total,
      page,
      limit,
      totalPages: Math.ceil(total / limit),
      data: files,
    })
  } catch (err) {
    console.error(err)
    res.status(500).send(err.message)
  }
}



const { processFileContent } = require("../services/vectorService");

exports.uploadFile = async (req, res) => {
  try {
    if (!req.file) return res.status(400).send("No file uploaded.");
    const filename = Buffer.from(req.file.originalname, "latin1").toString(
      "utf8"
    );
    console.log('file đường dẫn là:',req.file.path)
    
    const documentType = req.body.documentType || "Chưa xác định";
    // Gọi sinh embedding và lưu vào pgvector
    const { text, pgId } = await processFileContent(req.file.path);
    console.log("Text:", text);
    console.log("pgId:", pgId);
    const ext = path.extname(req.file.originalname).slice(1).toLowerCase();
// "docx"

    const fileData = {
      name: filename,
      size: req.file.size,
      type: ext,
      path: req.file.path,
      extractedText: text || null, // tạm
      embeddingId: pgId,
      tags: [],
      uploadedBy: null,
      documentType,
    };

    const savedFile = await File.create(fileData);

    
    // Hoặc pass extracted text nếu có: await processFileContent(req.body.extractedText);

    res.json(savedFile);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
};

exports.getFileById = async (req, res) => {
  try {
    const file = await File.findById(req.params.id);
    if (!file) return res.status(404).send("Không tìm thấy file.");
    res.json(file);
  } catch (err) {
    console.error(err);
    res.status(500).send(err.message);
  }
};
exports.updateFile = async (req, res) => {
  try {
    const file = await File.findById(req.params.id);
    if (!file) return res.status(404).send("Không tìm thấy file.");

    if (req.body.name) file.name = req.body.name;
    if (req.body.documentType) file.documentType = req.body.documentType;
    if (req.body.tags) file.tags = req.body.tags;
    if (req.body.extractedText) file.extractedText = req.body.extractedText;
    if (req.body.embeddingId) file.embeddingId = req.body.embeddingId;

    await file.save();
    res.json(file);
  } catch (err) {
    console.error(err);
    res.status(500).send(err.message);
  }
};
exports.deleteFile = async (req, res) => {
  try {
    const file = await File.findById(req.params.id);
    if (!file) return res.status(404).send("Không tìm thấy file.");

    // Không cần xóa trên ổ cứng nếu chỉ đánh dấu xóa logic

    // Xóa vector embedding nếu có
    if (file.embeddingId) {
      await pool.query("DELETE FROM documents WHERE id = $1", [file.embeddingId]);
    }

    // Đánh dấu đã xóa trong MongoDB
    file.deleted = true;
    await file.save();

    res.send("Đã đánh dấu xóa thành công.");
  } catch (err) {
    console.error(err);
    res.status(500).send(err.message);
  }
};

// exports.deleteFile = async (req, res) => {
//   try {
//     const file = await File.findById(req.params.id);
//     if (!file) return res.status(404).send("Không tìm thấy file.");

//     // Xóa file thật trên ổ cứng
//     // if (fs.existsSync(file.path)) {
//     //   fs.unlinkSync(file.path);
//     // }
//     if (file.embeddingId) {
//         await pool.query("DELETE FROM documents WHERE id = $1", [file.embeddingId]);
//     }
//     // Xóa trong MongoDB
//     await File.deleteOne({ _id: req.params.id });

//     res.send("Xóa thành công.");
//   } catch (err) {
//     console.error(err);
//     res.status(500).send(err.message);
//   }
// };
// 📁 fileController.js (bổ sung hai API sau)

// Thống kê số file đã tải lên, loại file, và file đã xóa
exports.getFileStats = async (req, res) => {
  try {
    const allFiles = await File.find().select('type createdAt deleted')

    const totalFiles = allFiles.length
    const deletedFiles = allFiles.filter(f => f.deleted).length

    const fileTypes = { pdf: 0, docx: 0, image: 0 }
    const daily = {}
    const weekly = {}
    const monthly = {}

    for (let file of allFiles) {
      if (!file.type || !file.createdAt) continue

      const type = normalizeType(file.type)
      
      if (!['pdf', 'docx', 'image'].includes(type)) continue

      const date = format(file.createdAt, 'yyyy-MM-dd')
      const month = format(file.createdAt, 'yyyy-MM')
      const week = `${getYear(file.createdAt)}-W${getISOWeek(file.createdAt)}`

      fileTypes[type] = (fileTypes[type] || 0) + 1

      // Daily
      daily[date] = daily[date] || {}
      daily[date][type] = (daily[date][type] || 0) + 1

      // Weekly
      weekly[week] = weekly[week] || {}
      weekly[week][type] = (weekly[week][type] || 0) + 1

      // Monthly
      monthly[month] = monthly[month] || {}
      monthly[month][type] = (monthly[month][type] || 0) + 1
    }

    const convert = (obj, keyLabel) =>
      Object.entries(obj)
        .sort(([a], [b]) => new Date(a) - new Date(b))
        .map(([key, data]) => ({ [keyLabel]: key, ...data }))

    res.json({
      totalFiles,
      deletedFiles,
      fileTypes,
      dailyStats: convert(daily, 'date'),
      weeklyStats: convert(weekly, 'week'),
      monthlyStats: convert(monthly, 'month'),
    })
  } catch (err) {
    console.error(err)
    res.status(500).send(err.message)
  }
}

// Danh sách file gần đây (limit 5)
exports.getRecentFiles = async (req, res) => {
  try {
    const files = await File.find()
      .sort({ createdAt: -1 })
      .limit(5)
      .populate('uploadedBy', 'username');

    res.json(files);
  } catch (err) {
    console.error(err);
    res.status(500).send(err.message);
  }
};
