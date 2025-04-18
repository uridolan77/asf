import React from 'react';
import { Fade, Grow, Slide, Zoom, Collapse } from '@mui/material';
import { styled } from '@mui/material/styles';

/**
 * Fade In animation component
 */
export const FadeIn = ({ children, ...props }) => {
  // Create a wrapper div to ensure we always have a valid DOM element
  const content = (
    <div style={{ width: '100%', height: '100%' }}>
      {children}
    </div>
  );

  return (
    <Fade in={true} mountOnEnter unmountOnExit {...props}>
      {content}
    </Fade>
  );
};

/**
 * Grow animation component
 */
export const GrowIn = ({ children, ...props }) => {
  // Create a wrapper div to ensure we always have a valid DOM element
  const content = (
    <div style={{ width: '100%', height: '100%' }}>
      {children}
    </div>
  );

  return (
    <Grow in={true} mountOnEnter unmountOnExit {...props}>
      {content}
    </Grow>
  );
};

/**
 * Slide In animation component
 */
export const SlideIn = ({ children, direction = 'right', ...props }) => {
  // Create a wrapper div to ensure we always have a valid DOM element
  const content = (
    <div style={{ width: '100%', height: '100%' }}>
      {children}
    </div>
  );

  return (
    <Slide direction={direction} in={true} mountOnEnter unmountOnExit {...props}>
      {content}
    </Slide>
  );
};

/**
 * Zoom In animation component
 */
export const ZoomIn = ({ children, ...props }) => {
  // Create a wrapper div to ensure we always have a valid DOM element
  const content = (
    <div style={{ width: '100%', height: '100%' }}>
      {children}
    </div>
  );

  return (
    <Zoom in={true} mountOnEnter unmountOnExit {...props}>
      {content}
    </Zoom>
  );
};

/**
 * Staggered animation for lists
 */
export const StaggeredList = ({ children, staggerDelay = 100, ...props }) => {
  return React.Children.map(children, (child, index) => {
    if (!React.isValidElement(child)) return child;

    return (
      <Fade
        in={true}
        style={{ transitionDelay: `${index * staggerDelay}ms` }}
        {...props}
      >
        {child}
      </Fade>
    );
  });
};

/**
 * Animated page transition
 */
export const PageTransition = ({ children, ...props }) => {
  // Create a wrapper div to ensure we always have a valid DOM element
  const content = (
    <div style={{ width: '100%', height: '100%' }}>
      {children}
    </div>
  );

  return (
    <Fade
      in={true}
      timeout={300}
      mountOnEnter
      unmountOnExit
      {...props}
    >
      {content}
    </Fade>
  );
};

/**
 * Hover animation wrapper
 */
const HoverAnimationWrapper = styled('div')(({ theme, scale = 1.03 }) => ({
  transition: theme.transitions.create(['transform', 'box-shadow'], {
    duration: theme.transitions.duration.shorter,
  }),
  '&:hover': {
    transform: `scale(${scale})`,
    boxShadow: theme.shadows[4],
  },
}));

export const HoverAnimation = ({ children, scale, ...props }) => {
  return (
    <HoverAnimationWrapper scale={scale} {...props}>
      {children}
    </HoverAnimationWrapper>
  );
};

/**
 * Animated counter
 */
export const AnimatedCounter = ({ value, duration = 1000 }) => {
  const [displayValue, setDisplayValue] = React.useState(0);

  React.useEffect(() => {
    let startTime;
    let animationFrame;
    const startValue = displayValue;
    const endValue = value;

    const updateValue = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      const currentValue = Math.floor(startValue + progress * (endValue - startValue));

      setDisplayValue(currentValue);

      if (progress < 1) {
        animationFrame = requestAnimationFrame(updateValue);
      }
    };

    animationFrame = requestAnimationFrame(updateValue);

    return () => {
      cancelAnimationFrame(animationFrame);
    };
  }, [value, duration, displayValue]);

  return <span>{displayValue}</span>;
};

export default {
  FadeIn,
  GrowIn,
  SlideIn,
  ZoomIn,
  StaggeredList,
  PageTransition,
  HoverAnimation,
  AnimatedCounter
};
