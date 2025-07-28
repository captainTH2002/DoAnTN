import { Card, CardContent } from "@/components/ui/card"

export default function DocumentCard({ name, type, date }) {
  return (
    <Card className="mb-4">
      <CardContent className="p-4">
        <h3 className="text-lg font-bold">{name}</h3>
        <p className="text-sm text-gray-500">Loại: {type}</p>
        <p className="text-sm text-gray-500">Ngày: {date}</p>
      </CardContent>
    </Card>
  )
}
