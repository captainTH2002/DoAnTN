import { useEffect, useState } from "react"
import Navbar from "@/components/ui/Navbar"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import axios from "../api/axiosUser"
import { FileText } from "lucide-react"

export default function Home() {
  const [docs, setDocs] = useState([])

  useEffect(() => {
    axios.get("/file")
      .then((res) => {
        setDocs(res.data?.data || [])
      })
      .catch((err) => {
        console.error("Lỗi khi lấy tài liệu:", err)
        setDocs([])
      })
  }, [])

  return (
    <>
      

      {/* Banner */}
      <section className="bg-gradient-to-r from-orange-200 via-orange-100 to-purple-100 py-12 px-6 text-center shadow-inner">
        <div className="max-w-4xl mx-auto space-y-3">
          <h1 className="text-4xl font-bold text-gray-800 tracking-tight">
            📚 Danh Sách Văn Bản Hành Chính
          </h1>
          <p className="text-lg text-gray-600">
            Hệ thống trích xuất và truy vấn văn bản hành chính
          </p>
        </div>
      </section>

      {/* Danh sách văn bản */}
      <main className="px-6 py-12 bg-gray-50 min-h-screen">
        <h2 className="text-2xl font-semibold text-gray-800 mb-6">
          📑 Xem danh sách các văn bản hành chính
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {docs.map((doc) => (
            <Card
              key={doc._id}
              className="transition-shadow hover:shadow-xl border border-gray-200 rounded-2xl"
            >
              <CardContent className="p-5 space-y-2">
                <div className="flex items-center gap-2 text-blue-600">
                  <FileText className="w-5 h-5" />
                  <h3 className="font-bold text-md truncate">{doc.filename}</h3>
                </div>

                <p className="text-sm text-gray-600">
                  🏛 Tên: <strong>{doc.name || "Không rõ"}</strong>
                </p>
                <p className="text-sm text-gray-600">
                  📅 Ngày ban hành:{" "}
                  {doc.createdAt ? new Date(doc.createdAt).toLocaleDateString() : "Không rõ"}
                </p>

                <Button
                  className="mt-3 w-full bg-gradient-to-r from-blue-500 to-indigo-500 text-white shadow hover:brightness-110 transition"
                >
                  Xem chi tiết
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </main>
    </>
  )
}
