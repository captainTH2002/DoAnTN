// src/contexts/AuthContext.jsx
import { createContext, useEffect, useState } from 'react';
import axios from "../api/axiosUser";
import { useNavigate } from 'react-router-dom';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  // Hàm kiểm tra người dùng từ token
  const fetchUser = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return null;

      const res = await axios.get('/users/me', {
        headers: { Authorization: `Bearer ${token}` },
      });
      return res.data;
    } catch (err) {
      console.log("❌ Fetch user failed:", err);
      return null;
    }
  };

  // Hàm đăng nhập
  const login = async (token) => {
    localStorage.setItem('token', token);
    const currentUser = await fetchUser();
    if (currentUser) {
      setUser(currentUser);
      navigate('/');
    } else {
      localStorage.removeItem('token');
    }
  };

  // Kiểm tra đăng nhập ban đầu
  useEffect(() => {
    const init = async () => {
      const currentUser = await fetchUser();
      if (currentUser) setUser(currentUser);
      setLoading(false);
    };
    init();
  }, []);

  return (
    <AuthContext.Provider value={{ user, setUser, loading, login }}>
      {!loading && children}
    </AuthContext.Provider>
  );
};

export { AuthContext };
