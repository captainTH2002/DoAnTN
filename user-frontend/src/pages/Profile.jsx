import { Card, CardContent } from "@/components/ui/card"
import Navbar from "@/components/ui/Navbar"

export default function Profile() {
  const user = JSON.parse(localStorage.getItem("user") || "{}")

  return (
    <>
      <main className="min-h-screen bg-gray-50 flex justify-center items-center p-4">
        <Card className="max-w-md w-full shadow-xl">
          <CardContent className="p-6 space-y-4">
            <h2 className="text-2xl font-bold text-primary">👤 Hồ sơ cá nhân</h2>
            <p><strong>Tên:</strong> {user.username}</p>
            <p><strong>Email:</strong> {user.email}</p>
            <button
              className="text-red-600 underline"
              onClick={() => {
                localStorage.clear()
                window.location.href = "/"
              }}
            >
              Đăng xuất
            </button>
          </CardContent>
        </Card>
      </main>
    </>
  )
}
