
// src/pages/Upload.jsx
import React, { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import axios from "../api/axiosUser"

export default function Upload() {
  const [file, setFile] = useState(null)
  const [message, setMessage] = useState("")

  const handleUpload = async () => {
    if (!file) return setMessage("❌ Vui lòng chọn tệp.")

    const formData = new FormData()
    formData.append("file", file)

    setMessage("⏳ Đang tải lên...")

    try {
      const res = await axios.post("/file", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      setMessage("✅ Tải lên thành công: " + res.data.filename)
    // eslint-disable-next-line no-unused-vars
    } catch (err) {
      setMessage("❌ Tải lên thất bại.")
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <Card className="w-full max-w-md shadow-xl">
        <CardContent className="space-y-4 p-6">
          <h1 className="text-2xl font-bold text-center text-primary">Tải lên tài liệu</h1>

          <Input type="file" onChange={(e) => setFile(e.target.files[0])} />

          <Button className="w-full bg-black text-white" onClick={handleUpload}>
            Tải lên
          </Button>

          {message && <p className="text-center text-sm text-gray-600">{message}</p>}
        </CardContent>
      </Card>
    </div>
  )
}
