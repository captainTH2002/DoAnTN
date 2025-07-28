import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import axios from "../api/axiosUser"

export default function RegisterPage() {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleRegister = async () => {
    try {
      await axios.post('/auth/register', { username, email, password });
      navigate('/login');
    } catch (err) {
      setError(err.response?.data?.error || 'Đăng ký thất bại');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-100 to-green-100">
      <Card className="w-full max-w-md shadow-2xl">
        <CardHeader>
          <CardTitle className="text-center text-2xl font-bold">Đăng Ký</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Input placeholder="Tên người dùng" value={username} onChange={(e) => setUsername(e.target.value)} />
          <Input placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
          <Input type="password" placeholder="Mật khẩu" value={password} onChange={(e) => setPassword(e.target.value)} />
          {error && <div className="text-red-500 text-sm">{error}</div>}
          <Button className="w-full" onClick={handleRegister}>Đăng ký</Button>
        </CardContent>
      </Card>
    </div>
  );
}
