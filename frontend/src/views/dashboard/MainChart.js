import React, { useEffect, useRef } from 'react'
import { CChartLine } from '@coreui/react-chartjs'
import { getStyle } from '@coreui/utils'

const COLORS = {
  pdf: getStyle('--cui-info'),
  docx: getStyle('--cui-success'),
  image: getStyle('--cui-warning'),
}

const MainChart = ({ data, labelKey = 'date' }) => {
  const chartRef = useRef(null)

  useEffect(() => {
    const updateTheme = () => {
      if (chartRef.current) {
        const options = chartRef.current.options
        options.scales.x.grid.borderColor = getStyle('--cui-border-color-translucent')
        options.scales.x.grid.color = getStyle('--cui-border-color-translucent')
        options.scales.x.ticks.color = getStyle('--cui-body-color')
        options.scales.y.grid.borderColor = getStyle('--cui-border-color-translucent')
        options.scales.y.grid.color = getStyle('--cui-border-color-translucent')
        options.scales.y.ticks.color = getStyle('--cui-body-color')
        chartRef.current.update()
      }
    }

    document.documentElement.addEventListener('ColorSchemeChange', updateTheme)
    return () => document.documentElement.removeEventListener('ColorSchemeChange', updateTheme)
  }, [])

  if (!data || data.length === 0) return null

  const labels = data.map((item) => item[labelKey])
  const types = ['pdf', 'docx', 'image']

  const datasets = types.map((type) => ({
    label: type.toUpperCase(),
    data: data.map((item) => item[type] || 0),
    borderColor: COLORS[type] || '#000',
    backgroundColor: 'transparent',
    tension: 0.4,
    fill: false,
    pointBackgroundColor: COLORS[type] || '#000',
  }))

  const maxValue = Math.max(...datasets.flatMap((ds) => ds.data), 5)

  return (
    <CChartLine
      ref={chartRef}
      style={{ height: '300px', marginTop: '40px' }}
      data={{ labels, datasets }}
      options={{
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true },
        },
        scales: {
          x: {
            type: 'category', // 👈 Fix chính
            title: { display: true, text: labelKey.toUpperCase() },
            grid: {
              color: getStyle('--cui-border-color-translucent'),
              drawOnChartArea: false,
            },
            ticks: {
              color: getStyle('--cui-body-color'),
              autoSkip: true,
              maxRotation: 45,
              minRotation: 0,
            },
          },
          y: {
            beginAtZero: true,
            max: maxValue + 5,
            grid: {
              color: getStyle('--cui-border-color-translucent'),
            },
            ticks: {
              color: getStyle('--cui-body-color'),
              maxTicksLimit: 5,
              stepSize: Math.ceil((maxValue + 5) / 5),
            },
            title: { display: true, text: 'Số lượng' },
          },
        },
        elements: {
          point: {
            radius: 3,
            hitRadius: 10,
            hoverRadius: 6,
            hoverBorderWidth: 3,
          },
        },
      }}
    />
  )
}

export default MainChart
