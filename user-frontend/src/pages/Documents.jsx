import { useEffect, useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import axios from "../api/axiosUser"
import Navbar from "@/components/ui/Navbar"

export default function Documents() {
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
      <main className="min-h-screen bg-gray-50 p-6">
        <h2 className="text-2xl font-bold mb-6 text-primary">📁 Danh sách tài liệu</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {docs.map((doc) => (
            <Card key={doc._id}>
              <CardContent className="p-4">
                <p className="font-semibold">{doc.name}</p>
                <p className="text-sm text-gray-500">{new Date(doc.createdAt).toLocaleString()}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </main>
    </>
  )
}
