export const formatLabelForDisplay = (label?: string | null): string => {
  if (!label) {
    return 'N/A';
  }

  const normalized = label.trim();
  if (!normalized) {
    return 'N/A';
  }

  const upper = normalized.toUpperCase();
  if (upper === 'N/A' || upper === 'NA') {
    return 'N/A';
  }

  const sanitized = normalized.replace(/[_]+/g, ' ').replace(/\s+/g, ' ');

  return sanitized
    .split(' ')
    .filter(Boolean)
    .map((word) => {
      const upperWord = word.toUpperCase();
      if (upperWord.length <= 3) {
        return upperWord;
      }
      const lower = word.toLowerCase();
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(' ');
};

export const normalizeLabel = (label?: string | null): string => {
  if (!label) {
    return 'N/A';
  }
  const trimmed = label.trim();
  if (!trimmed) {
    return 'N/A';
  }
  return trimmed.toUpperCase();
};
