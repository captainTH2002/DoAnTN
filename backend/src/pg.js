// src/pg.js
const { Pool } = require('pg');

const pool = new Pool({
  host: 'localhost',      // địa chỉ db là máy local
  port: 5432,             // cổng mặc định PostgreSQL
  user: 'postgres',
  password: '123456',
  database: 'vectordb'
});

module.exports = pool;

