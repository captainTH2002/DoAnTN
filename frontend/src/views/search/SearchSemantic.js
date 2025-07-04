import React, { useState } from 'react'
import axios from '../../api/axios'
import { CFormInput, CButton, CTable } from '@coreui/react'

const SearchSemantic = () => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])

  const handleSearch = async () => {
    if (!query) return
    try {
      const res = await axios.get(`/file/search?q=${encodeURIComponent(query)}`)
      setResults(res.data)
    } catch (err) {
      console.error('Error searching:', err)
      alert('Search failed')
    }
  }

  const columns = [
    { key: 'id', label: 'ID', _props: { scope: 'col' } },
    { key: 'content', label: 'Content', _props: { scope: 'col' } }
  ]

  return (
    <div>
      <h3>Semantic Search</h3>
      <div className="d-flex mb-3">
        <CFormInput
          placeholder="Nhập từ khóa..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="me-2"
        />
        <CButton onClick={handleSearch}>Tìm kiếm</CButton>
      </div>

      {results.length > 0 && (
        <CTable columns={columns} items={results} className="table-striped table-bordered" />
      )}
    </div>
  )
}

export default SearchSemantic
