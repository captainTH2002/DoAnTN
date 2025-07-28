import { useState } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import axios from "../api/axiosUser"
import useAuth from '../context/useAuth'; 

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { login } = useAuth();
  const handleLogin = async () => {
    try {
      const res = await axios.post('/auth/login', { email, password });
      localStorage.setItem('token', res.data.token);
      login(res.data.token, res.data.user)
      console.log(res.data.token)
    } catch (err) {
      setError(err.response?.data?.error || 'Đăng nhập thất bại');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-100 to-blue-100">
      <Card className="w-full max-w-md shadow-2xl">
        <CardHeader>
          <CardTitle className="text-center text-2xl font-bold">Đăng Nhập</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Input placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
          <Input type="password" placeholder="Mật khẩu" value={password} onChange={(e) => setPassword(e.target.value)} />
          {error && <div className="text-red-500 text-sm">{error}</div>}
          <Button className="w-full" onClick={handleLogin}>Đăng nhập</Button>
        </CardContent>
      </Card>
    </div>
  );
}
