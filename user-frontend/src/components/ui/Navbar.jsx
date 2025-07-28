import { Button } from "@/components/ui/button"
import { Link } from "react-router-dom"

export default function Navbar() {
  return (
    <header className="w-full border-b bg-white shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center">
        <Link to="/" className="text-xl font-bold text-gray-800 hover:opacity-80 transition">
          📁 Văn bản hành chính
        </Link>

        <div className="space-x-2">
          <Link to="/upload">
            <Button className="bg-gray-100 text-gray-800 hover:bg-gray-200 shadow-sm transition duration-200">
              Tải lên
            </Button>
          </Link>

          <Link to="/ask">
            <Button className="bg-blue-100 text-blue-800 hover:bg-blue-200 shadow-sm transition duration-200">
              Hỏi đáp
            </Button>
          </Link>

          <Link to="/documents">
            <Button className="bg-green-100 text-green-800 hover:bg-green-200 shadow-sm transition duration-200">
              Tài liệu
            </Button>
          </Link>

          <Link to="/profile">
            <Button className="bg-purple-100 text-purple-800 hover:bg-purple-200 shadow-sm transition duration-200">
              Tài khoản
            </Button>
          </Link>
        </div>
      </div>
    </header>
  )
}
