#include "starfield.h"
#include <QPainter>
#include <QRandomGenerator>
#include <QDateTime>
#include <QResizeEvent>
#include <QPaintEvent>
#include <QShowEvent>
#include <QHideEvent>
#include <QRadialGradient>
#include <QPropertyAnimation>
#include <QParallelAnimationGroup>
#include <QEasingCurve>
#include <QtMath>

// Define star colors (different stellar classifications)
const QVector<QColor> StarField::STAR_COLORS = {
    QColor(255, 255, 255),    // White (Sun-like)
    QColor(255, 244, 234),    // Warm white
    QColor(255, 209, 178),    // Orange
    QColor(255, 180, 107),    // Yellow-orange
    QColor(170, 191, 255),    // Blue-white
    QColor(155, 176, 255),    // Blue
    QColor(255, 204, 111),    // Yellow
    QColor(255, 166, 81)      // Orange-red
};

StarField::StarField(QWidget *parent)
    : QWidget(parent)
    , m_animationTimer(new QTimer(this))
    , m_progressAnimation(nullptr)
    , m_animationGroup(nullptr)
    , m_starCount(DEFAULT_STAR_COUNT)
    , m_animationProgress(0.0f)
    , m_isAnimating(false)
    , m_isPaused(false)
    , m_lastUpdateTime(0)
    , m_deltaTime(0.0f)
    , m_animationSpeed(DEFAULT_ANIMATION_SPEED)
    , m_twinkleEnabled(true)
    , m_movementEnabled(true)
    , m_colorVariation(true)
{
    // Set widget properties
    setAttribute(Qt::WA_TransparentForMouseEvents);
    setAttribute(Qt::WA_NoSystemBackground);
    setAutoFillBackground(false);
    
    // Initialize random seed
    QRandomGenerator::global()->seed(QDateTime::currentMSecsSinceEpoch());
    
    // Setup animation timer
    m_animationTimer->setInterval(ANIMATION_INTERVAL);
    connect(m_animationTimer, &QTimer::timeout, this, &StarField::updateAnimation);
    
    // Initialize stars
    initializeStars();
    
    // Setup progress animation
    m_progressAnimation = new QPropertyAnimation(this, "animationProgress", this);
    m_progressAnimation->setDuration(2000);
    m_progressAnimation->setEasingCurve(QEasingCurve::InOutQuad);
    connect(m_progressAnimation, &QPropertyAnimation::finished, 
            this, &StarField::onAnimationFinished);
    
    m_lastUpdateTime = QDateTime::currentMSecsSinceEpoch();
}

StarField::~StarField()
{
    stopAnimation();
}

void StarField::initializeStars()
{
    m_stars.clear();
    m_stars.reserve(m_starCount);
    
    for (int i = 0; i < m_starCount; ++i) {
        Star star;
        generateRandomStar(star);
        m_stars.append(star);
    }
}

void StarField::generateRandomStar(Star &star)
{
    auto *rng = QRandomGenerator::global();
    
    // Random position
    star.position.setX(rng->bounded(width()));
    star.position.setY(rng->bounded(height()));
    
    // Random velocity (slow drift)
    float angle = rng->bounded(360) * M_PI / 180.0f;
    float speed = rng->bounded(MIN_VELOCITY * 100, MAX_VELOCITY * 100) / 100.0f;
    star.velocity.setX(cos(angle) * speed);
    star.velocity.setY(sin(angle) * speed);
    
    // Random size and brightness
    star.size = rng->bounded(MIN_STAR_SIZE * 100, MAX_STAR_SIZE * 100) / 100.0f;
    star.brightness = rng->bounded(MIN_BRIGHTNESS * 100, MAX_BRIGHTNESS * 100) / 100.0f;
    
    // Random twinkle properties
    star.twinklePhase = rng->bounded(360) * M_PI / 180.0f;
    star.twinkleSpeed = rng->bounded(TWINKLE_SPEED_MIN * 100, TWINKLE_SPEED_MAX * 100) / 100.0f;
    
    // Random color
    star.color = generateStarColor();
}

QColor StarField::generateStarColor()
{
    if (!m_colorVariation) {
        return QColor(255, 255, 255);
    }
    
    auto *rng = QRandomGenerator::global();
    
    // 70% chance for white/warm white stars
    if (rng->bounded(100) < 70) {
        return STAR_COLORS[rng->bounded(2)];
    }
    
    // 30% chance for colored stars
    return STAR_COLORS[rng->bounded(STAR_COLORS.size())];
}

void StarField::startAnimation()
{
    if (m_isAnimating) {
        return;
    }
    
    m_isAnimating = true;
    m_isPaused = false;
    m_lastUpdateTime = QDateTime::currentMSecsSinceEpoch();
    
    m_animationTimer->start();
    
    // Start progress animation
    if (m_progressAnimation) {
        m_progressAnimation->setStartValue(0.0f);
        m_progressAnimation->setEndValue(1.0f);
        m_progressAnimation->start();
    }
}

void StarField::stopAnimation()
{
    m_isAnimating = false;
    m_isPaused = false;
    
    m_animationTimer->stop();
    
    if (m_progressAnimation) {
        m_progressAnimation->stop();
    }
    
    m_animationProgress = 0.0f;
}

void StarField::pauseAnimation()
{
    if (!m_isAnimating || m_isPaused) {
        return;
    }
    
    m_isPaused = true;
    m_animationTimer->stop();
    
    if (m_progressAnimation) {
        m_progressAnimation->pause();
    }
}

void StarField::resumeAnimation()
{
    if (!m_isAnimating || !m_isPaused) {
        return;
    }
    
    m_isPaused = false;
    m_lastUpdateTime = QDateTime::currentMSecsSinceEpoch();
    m_animationTimer->start();
    
    if (m_progressAnimation) {
        m_progressAnimation->resume();
    }
}

void StarField::setAnimationProgress(float progress)
{
    m_animationProgress = qBound(0.0f, progress, 1.0f);
    update();
}

void StarField::setStarCount(int count)
{
    if (count != m_starCount && count > 0) {
        m_starCount = count;
        initializeStars();
        update();
    }
}

void StarField::setAnimationSpeed(float speed)
{
    m_animationSpeed = qMax(0.1f, speed);
}

void StarField::setTwinkleEnabled(bool enabled)
{
    m_twinkleEnabled = enabled;
}

void StarField::setMovementEnabled(bool enabled)
{
    m_movementEnabled = enabled;
}

void StarField::setColorVariation(bool enabled)
{
    if (enabled != m_colorVariation) {
        m_colorVariation = enabled;
        // Regenerate star colors
        for (Star &star : m_stars) {
            star.color = generateStarColor();
        }
        update();
    }
}

void StarField::updateAnimation()
{
    if (!m_isAnimating || m_isPaused) {
        return;
    }
    
    // Calculate delta time
    qint64 currentTime = QDateTime::currentMSecsSinceEpoch();
    m_deltaTime = (currentTime - m_lastUpdateTime) / 1000.0f;
    m_lastUpdateTime = currentTime;
    
    updateStars();
    update();
}

void StarField::updateStars()
{
    for (Star &star : m_stars) {
        updateStarPosition(star);
        updateStarTwinkle(star);
    }
}

void StarField::updateStarPosition(Star &star)
{
    if (!m_movementEnabled) {
        return;
    }
    
    // Update position based on velocity
    star.position += star.velocity * m_deltaTime * m_animationSpeed;
    
    // Wrap around screen edges
    if (star.position.x() < -10) {
        star.position.setX(width() + 10);
    } else if (star.position.x() > width() + 10) {
        star.position.setX(-10);
    }
    
    if (star.position.y() < -10) {
        star.position.setY(height() + 10);
    } else if (star.position.y() > height() + 10) {
        star.position.setY(-10);
    }
}

void StarField::updateStarTwinkle(Star &star)
{
    if (!m_twinkleEnabled) {
        return;
    }
    
    // Update twinkle phase
    star.twinklePhase += star.twinkleSpeed * m_deltaTime * m_animationSpeed;
    
    // Keep phase in reasonable range
    if (star.twinklePhase > 2 * M_PI) {
        star.twinklePhase -= 2 * M_PI;
    }
}

void StarField::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event)
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect(), QColor(15, 15, 23)); // Deep space background
    
    drawStars(painter);
}

void StarField::drawStars(QPainter &painter)
{
    for (const Star &star : m_stars) {
        drawStar(painter, star);
    }
}

void StarField::drawStar(QPainter &painter, const Star &star)
{
    // Calculate current brightness with twinkle effect
    float currentBrightness = star.brightness;
    if (m_twinkleEnabled) {
        float twinkle = (sin(star.twinklePhase) + 1.0f) * 0.5f; // 0.0 to 1.0
        currentBrightness *= (0.7f + 0.3f * twinkle); // Vary between 70% and 100%
    }
    
    // Apply animation progress fade-in
    currentBrightness *= m_animationProgress;
    
    if (currentBrightness <= 0.01f) {
        return; // Don't draw nearly invisible stars
    }
    
    // Calculate alpha
    int alpha = static_cast<int>(255 * currentBrightness);
    
    // Set color with calculated alpha
    QColor starColor = star.color;
    starColor.setAlpha(alpha);
    
    // Calculate star size
    float currentSize = star.size * (0.8f + 0.2f * currentBrightness);
    
    // Draw star with glow effect for brighter stars
    if (currentBrightness > 0.7f && currentSize > 1.5f) {
        // Draw glow
        QRadialGradient gradient(star.position, currentSize * 2);
        QColor glowColor = starColor;
        glowColor.setAlpha(alpha / 4);
        gradient.setColorAt(0, starColor);
        gradient.setColorAt(0.3, glowColor);
        gradient.setColorAt(1, Qt::transparent);
        
        painter.setBrush(QBrush(gradient));
        painter.setPen(Qt::NoPen);
        painter.drawEllipse(star.position, currentSize * 2, currentSize * 2);
    }
    
    // Draw main star
    painter.setBrush(QBrush(starColor));
    painter.setPen(Qt::NoPen);
    
    if (currentSize < 1.0f) {
        // Draw as point for small stars
        painter.drawPoint(star.position);
    } else {
        // Draw as circle for larger stars
        painter.drawEllipse(star.position, currentSize, currentSize);
        
        // Add cross pattern for very bright stars
        if (currentBrightness > 0.8f && currentSize > 2.0f) {
            QPen crossPen(starColor, 0.5);
            painter.setPen(crossPen);
            
            float crossSize = currentSize * 1.5f;
            painter.drawLine(star.position.x() - crossSize, star.position.y(),
                           star.position.x() + crossSize, star.position.y());
            painter.drawLine(star.position.x(), star.position.y() - crossSize,
                           star.position.x(), star.position.y() + crossSize);
        }
    }
}

void StarField::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    
    // Redistribute stars for new size
    if (event->size() != event->oldSize()) {
        for (Star &star : m_stars) {
            // Keep stars proportionally positioned
            if (event->oldSize().width() > 0 && event->oldSize().height() > 0) {
                float xRatio = static_cast<float>(event->size().width()) / event->oldSize().width();
                float yRatio = static_cast<float>(event->size().height()) / event->oldSize().height();
                
                star.position.setX(star.position.x() * xRatio);
                star.position.setY(star.position.y() * yRatio);
            }
        }
    }
}

void StarField::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);
    
    if (!m_isAnimating) {
        startAnimation();
    }
}

void StarField::hideEvent(QHideEvent *event)
{
    QWidget::hideEvent(event);
    
    if (m_isAnimating) {
        pauseAnimation();
    }
}

void StarField::onAnimationFinished()
{
    // Animation completed, continue with steady state
    if (m_progressAnimation) {
        m_progressAnimation->setStartValue(1.0f);
        m_progressAnimation->setEndValue(1.0f);
    }
}
