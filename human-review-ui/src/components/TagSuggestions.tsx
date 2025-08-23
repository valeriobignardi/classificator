import React, { useState, useEffect } from 'react';
import {
  Box,
  Chip,
  TextField,
  Typography,
  Badge,
  Tooltip,
  InputAdornment,
  IconButton
} from '@mui/material';
import {
  Search as SearchIcon,
  Clear as ClearIcon,
  Label as LabelIcon,
  AutoAwesome as AutoAwesomeIcon,
  Person as PersonIcon
} from '@mui/icons-material';

interface Tag {
  tag: string;
  count: number;
  source: 'automatic' | 'human_review' | 'mixed';
  avg_confidence: number;
}

interface TagSuggestionsProps {
  tags: Tag[];
  onTagSelect: (tag: string) => void;
  currentValue?: string;
  loading?: boolean;
}

const TagSuggestions: React.FC<TagSuggestionsProps> = ({
  tags,
  onTagSelect,
  currentValue = '',
  loading = false
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredTags, setFilteredTags] = useState<Tag[]>(tags);

  useEffect(() => {
    // Filtra i tag basandosi sul termine di ricerca
    const filtered = tags.filter(tag =>
      tag && tag.tag && tag.tag.toLowerCase().includes(searchTerm.toLowerCase())
    );
    
    // Ordina per rilevanza: prima i tag che iniziano con il termine di ricerca,
    // poi per frequenza d'uso
    filtered.sort((a, b) => {
      const aTag = a.tag || '';
      const bTag = b.tag || '';
      const aStartsWith = aTag.toLowerCase().startsWith(searchTerm.toLowerCase());
      const bStartsWith = bTag.toLowerCase().startsWith(searchTerm.toLowerCase());
      
      if (aStartsWith && !bStartsWith) return -1;
      if (!aStartsWith && bStartsWith) return 1;
      
      // Se entrambi iniziano o non iniziano con il termine, ordina per conteggio
      return (b.count || 0) - (a.count || 0);
    });
    
    setFilteredTags(filtered);
  }, [searchTerm, tags]);

  const getSourceIcon = (source: string) => {
    switch (source) {
      case 'automatic':
        return <AutoAwesomeIcon sx={{ fontSize: 14 }} />;
      case 'human_review':
        return <PersonIcon sx={{ fontSize: 14 }} />;
      case 'mixed':
        return <LabelIcon sx={{ fontSize: 14 }} />;
      default:
        return null;
    }
  };

  const getSourceColor = (source: string) => {
    switch (source) {
      case 'automatic':
        return 'primary';
      case 'human_review':
        return 'secondary';
      case 'mixed':
        return 'success';
      default:
        return 'default';
    }
  };

  const getSourceLabel = (source: string) => {
    switch (source) {
      case 'automatic':
        return 'Classificazione automatica';
      case 'human_review':
        return 'Revisione umana';
      case 'mixed':
        return 'Fonte mista';
      default:
        return 'Sconosciuto';
    }
  };

  const handleTagClick = (tag: string) => {
    onTagSelect(tag);
    setSearchTerm(''); // Pulisce la ricerca dopo la selezione
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <LabelIcon color="primary" />
        Tag Suggeriti
        <Badge badgeContent={filteredTags.length} color="primary" />
      </Typography>

      {/* Campo di ricerca */}
      <TextField
        fullWidth
        size="small"
        variant="outlined"
        placeholder="Cerca tag esistenti..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon color="action" />
            </InputAdornment>
          ),
          endAdornment: searchTerm && (
            <InputAdornment position="end">
              <IconButton
                size="small"
                onClick={() => setSearchTerm('')}
                edge="end"
              >
                <ClearIcon />
              </IconButton>
            </InputAdornment>
          )
        }}
        sx={{ mb: 2 }}
        disabled={loading}
      />

      {/* Lista dei tag */}
      <Box 
        sx={{ 
          maxHeight: 300, 
          overflow: 'auto',
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 1,
          p: 1,
          backgroundColor: 'background.paper'
        }}
      >
        {loading ? (
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
            Caricamento tag...
          </Typography>
        ) : filteredTags.length === 0 ? (
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
            {searchTerm ? 'Nessun tag trovato' : 'Nessun tag disponibile'}
          </Typography>
        ) : (
          <Box display="flex" flexWrap="wrap" gap={1}>
            {filteredTags.map((tagItem) => (
              <Tooltip
                key={tagItem.tag || 'unknown'}
                title={
                  <Box>
                    <Typography variant="body2">
                      <strong>{tagItem.tag || 'N/A'}</strong>
                    </Typography>
                    <Typography variant="caption">
                      Utilizzato {tagItem.count || 0} volte
                    </Typography>
                    <Typography variant="caption" display="block">
                      {getSourceLabel(tagItem.source || 'automatic')}
                    </Typography>
                    <Typography variant="caption" display="block">
                      Confidenza media: {((tagItem.avg_confidence || 0) * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                }
                arrow
              >
                <Chip
                  label={tagItem.tag || 'N/A'}
                  icon={getSourceIcon(tagItem.source || 'automatic') || undefined}
                  onClick={() => handleTagClick(tagItem.tag || '')}
                  variant={currentValue === tagItem.tag ? "filled" : "outlined"}
                  color={getSourceColor(tagItem.source || 'automatic') as any}
                  size="small"
                  sx={{
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    '&:hover': {
                      transform: 'scale(1.05)',
                      boxShadow: 2
                    },
                    ...(currentValue === tagItem.tag && {
                      boxShadow: 2,
                      fontWeight: 'bold'
                    })
                  }}
                />
              </Tooltip>
            ))}
          </Box>
        )}
      </Box>

      {/* Legenda */}
      <Box sx={{ mt: 2, p: 1, backgroundColor: 'grey.50', borderRadius: 1 }}>
        <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
          Legenda fonti:
        </Typography>
        <Box display="flex" gap={2} flexWrap="wrap">
          <Box display="flex" alignItems="center" gap={0.5}>
            <AutoAwesomeIcon sx={{ fontSize: 14, color: 'primary.main' }} />
            <Typography variant="caption">Automatico</Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={0.5}>
            <PersonIcon sx={{ fontSize: 14, color: 'secondary.main' }} />
            <Typography variant="caption">Umano</Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={0.5}>
            <LabelIcon sx={{ fontSize: 14, color: 'success.main' }} />
            <Typography variant="caption">Misto</Typography>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default TagSuggestions;
