const File = require("../models/File");
const path = require("path");
const pool = require("../pg.js");

exports.getFiles = async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const skip = (page - 1) * limit;

    const total = await File.countDocuments();
    const files = await File.find()
      .skip(skip)
      .limit(limit)
      .sort({ createdAt: -1 });
    
    res.json({
      total,
      page,
      limit,
      totalPages: Math.ceil(total / limit),
      data: files,
    });
  } catch (err) {
    console.error(err);
    res.status(500).send(err.message);
  }
};

const { processFileContent } = require("../services/vectorService");

exports.uploadFile = async (req, res) => {
  try {
    if (!req.file) return res.status(400).send("No file uploaded.");

    const filename = Buffer.from(req.file.originalname, "latin1").toString(
      "utf8"
    );
    const path = Buffer.from(req.file.path, "latin1").toString(
      "utf8"
    );
    const documentType = req.body.documentType || "Chưa xác định";
    // Gọi sinh embedding và lưu vào pgvector
    const { text, pgId } = await processFileContent(req.file.path);
    console.log("Text:", text);
    console.log("pgId:", pgId);
      
    const fileData = {
      name: filename,
      size: req.file.size,
      type: req.file.mimetype,
      path: path,
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

    // Xóa file thật trên ổ cứng
    // if (fs.existsSync(file.path)) {
    //   fs.unlinkSync(file.path);
    // }
    if (file.embeddingId) {
        await pool.query("DELETE FROM documents WHERE id = $1", [file.embeddingId]);
    }
    // Xóa trong MongoDB
    await File.deleteOne({ _id: req.params.id });

    res.send("Xóa thành công.");
  } catch (err) {
    console.error(err);
    res.status(500).send(err.message);
  }
};