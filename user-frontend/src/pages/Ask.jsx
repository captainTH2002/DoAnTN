import React, { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import axios from "../api/axiosUser";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm"; // nhớ cài: npm install remark-gfm
import { motion } from "framer-motion";
import { Paperclip, SendHorizonal } from "lucide-react"; // Icon tải lên & gửi
export default function Ask() {
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  // Tải danh sách cuộc hội thoại khi load trang
  useEffect(() => {
    fetchConversations();
  }, []);

  // Lấy danh sách các cuộc hội thoại
  const fetchConversations = async () => {
    const token = localStorage.getItem("token");
    console.log(token);
    const res = await axios.get("/conversations");
    setConversations(res.data);
  };

  // Lấy tin nhắn của một cuộc hội thoại
  const loadMessages = async (conversationId) => {
    setCurrentConversationId(conversationId);
    const res = await axios.get(`/messages/${conversationId}`);

    setMessages(res.data);
    console.log("Messages:", res.data);
  };

  // Tạo cuộc hội thoại mới
  const createNewConversation = async () => {
    const res = await axios.post("/conversations");
    const newConv = res.data;
    setConversations([newConv, ...conversations]);
    setCurrentConversationId(newConv._id);
    setMessages([]);
  };

  // Gửi câu hỏi trong cuộc hội thoại hiện tại
  const handleAsk = async () => {
    if (!question.trim() || !currentConversationId) return;
    setLoading(true);

    // Hiển thị tạm tin nhắn người dùng
    const newMessages = [...messages, { role: "user", content: question }];
    setMessages(newMessages);

    try {
      const res = await axios.post(`/chat/${currentConversationId}`, {
        question,
      });

      setMessages([
        ...newMessages,
        { role: "bot", content: res.data.answer || "Không có câu trả lời." },
      ]);
      setQuestion("");
      // eslint-disable-next-line no-unused-vars
    } catch (err) {
      setMessages([
        ...newMessages,
        { role: "bot", content: "❌ Lỗi khi gửi câu hỏi." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file) => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("/file", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const docInfo = res.data;
      setMessages([
        ...messages,
        {
          role: "user",
          content: `📎 Đã tải lên tài liệu: **${
            docInfo.filename || file.name
          }**`,
        },
        {
          role: "bot",
          content:
            "✅ Tài liệu đã được xử lý. Bạn có thể đặt câu hỏi liên quan đến nội dung.",
        },
      ]);
    } catch (err) {
      console.error("Upload failed", err);
      setMessages([
        ...messages,
        {
          role: "bot",
          content: "❌ Tải tài liệu thất bại.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-gray-100">
      {/* Sidebar */}
      {/* Sidebar trái giống UI hình ảnh */}
<aside className="w-72 bg-white border-r shadow-sm flex flex-col p-4">
  {/* Tiêu đề */}
  <div className="flex items-center justify-between mb-6">
    <div className="flex items-center gap-3">
      <div className="w-10 h-10 bg-purple-100 text-purple-600 font-bold rounded-full flex items-center justify-center">
        🧾
      </div>
      <div>
        <div className="text-base font-semibold text-gray-800">Câu Hỏi Mới</div>
      </div>
    </div>
    <Button
      size="icon"
      variant="ghost"
      onClick={createNewConversation}
      className="hover:bg-primary hover:text-white transition-all"
    >
      <span className="text-xl">＋</span>
    </Button>
  </div>

  {/* Danh sách hội thoại */}
  <ScrollArea className="flex-1 pr-1">
    <div className="space-y-2">
      {conversations.map((conv) => (
        <motion.div
          key={conv._id}
          whileHover={{ scale: 1.01 }}
          onClick={() => loadMessages(conv._id)}
          className={`cursor-pointer p-3 rounded-lg border text-sm leading-snug transition-all duration-200 ${
            conv._id === currentConversationId
              ? "bg-blue-100 border-blue-300 text-blue-800"
              : "bg-gray-50 hover:bg-gray-100 border-gray-200 text-gray-700"
          }`}
        >
          <div className="font-medium truncate">{conv.title || "Không tiêu đề"}</div>
          <div className="text-xs text-gray-500 mt-1">
            {new Date(conv.createdAt).toLocaleDateString("vi-VN")}
          </div>
        </motion.div>
      ))}
    </div>
  </ScrollArea>

  {/* Footer nhỏ (tùy chọn) */}
  {/* <div className="mt-4 text-xs text-gray-400 text-center">
    © CTU Linguitech - 2025
  </div> */}
</aside>


      {/* Main */}
      <main className="flex-1 flex flex-col items-center justify-start px-4 py-6 space-y-6">
        {currentConversationId ? (
          <>
            <div className="w-full max-w-2xl space-y-4">
              <Card>
                <CardContent className="p-4 space-y-4">
                  {messages.map((msg, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.2 }}
                      className={`rounded-lg p-3 shadow-sm border ${
                        msg.role === "user"
                          ? "bg-blue-50 border-blue-200"
                          : "bg-green-50 border-green-200"
                      }`}
                    >
                      <div
                        className={`text-sm font-semibold mb-2 ${
                          msg.role === "user"
                            ? "text-blue-800"
                            : "text-green-800"
                        }`}
                      >
                        {msg.role === "user" ? "🧑 Bạn:" : "🤖 Trợ lý:"}
                      </div>
                      <motion.div className="prose prose-sm max-w-none whitespace-pre-wrap break-words">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                      </motion.div>
                    </motion.div>
                  ))}
                </CardContent>
              </Card>

              <div className="flex items-center space-x-2 w-full max-w-2xl">
                <div className="relative flex-1">
                  <Input
                    placeholder="Nhập câu hỏi..."
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    disabled={loading}
                    className="pr-10"
                  />
                  {/* Nút icon 📎 */}
                  <label className="absolute right-2 top-1/2 transform -translate-y-1/2 cursor-pointer text-gray-500 hover:text-black">
                    <Paperclip size={18} />
                    <input
                      type="file"
                      onChange={(e) => handleFileUpload(e.target.files[0])}
                      className="hidden"
                    />
                  </label>
                </div>

                <Button
                  onClick={handleAsk}
                  disabled={loading}
                  className="gap-2"
                >
                  <SendHorizonal size={16} />
                  Gửi
                </Button>
              </div>
            </div>
          </>
        ) : (
          <p className="text-gray-500 text-lg">
            🧾 Hãy chọn hoặc tạo một cuộc hội thoại.
          </p>
        )}
      </main>
    </div>
  );
}
