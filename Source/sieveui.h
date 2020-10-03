//******************************************************************************
// Copyright 2020 ThirtySomething
//******************************************************************************
// This file is part of Sieve.
//
// Sieve is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// Sieve is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
// more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with Sieve. If not, see <http://www.gnu.org/licenses/>.
//******************************************************************************

#ifndef SIEVEUI_H
#define SIEVEUI_H

#include "csieve.h"
#include <QLabel>
#include <QMainWindow>
#include <QStatusBar>
#include <future>

QT_BEGIN_NAMESPACE
namespace Ui
{
    class SieveUI;
}
QT_END_NAMESPACE

/// <summary>
/// Class for UI of sieve
/// </summary>
class SieveUI : public QMainWindow
{
    Q_OBJECT

public:
    /// <summary>
    /// Default QT constructor
    /// </summary>
    /// <param name="parent">No parent window given</param>
    SieveUI(QWidget *parent = nullptr);

    /// <summary>
    /// Default destructor
    /// </summary>
    ~SieveUI(void);

signals:
    /// <summary>
    /// Signal is triggered when sieve found new prime
    /// </summary>
    /// <param name="newPrime">New prime</param>
    void primeChanged(long long newPrime);

private slots:
    /// <summary>
    /// Displays about box
    /// </summary>
    void on_actionAbout_Sieve_triggered(void);

    /// <summary>
    /// Export dialog to export primes
    /// </summary>
    void on_actionExport_triggered(void);

    /// <summary>
    /// Load dialog to load sieve
    /// </summary>
    void on_actionLoad_triggered(void);

    /// <summary>
    /// New dialog to create new sieve
    /// </summary>
    void on_actionNew_triggered(void);

    /// <summary>
    /// To exit the application
    /// </summary>
    void on_actionQuit_triggered(void);

    /// <summary>
    /// Save dialog to save sieve
    /// </summary>
    void on_actionSave_triggered(void);

    /// <summary>
    /// Start/resume sieving process
    /// </summary>
    void on_btnStart_clicked(void);

    /// <summary>
    /// Stop sieving process
    /// </summary>
    void on_btnStop_clicked(void);

    /// <summary>
    /// Called internally to update latest prime
    /// </summary>
    /// <param name="prime"></param>
    void setPrime(long long prime);

private:
    /// <summary>
    /// Initialize UI elements
    /// </summary>
    void initQtElements(void);

    /// <summary>
    /// Handle for running the sieve process as thread
    /// </summary>
    std::future<void> m_processSieve;

    /// <summary>
    /// The sieve algorithm
    /// </summary>
    std::unique_ptr<net::derpaul::sieve::CSieve> m_sieve;

    /// <summary>
    /// Statusbar of main windows
    /// </summary>
    QStatusBar *m_statusBar;

    /// <summary>
    /// Pointer to QT ui object
    /// </summary>
    Ui::SieveUI *ui;
};
#endif // SIEVEUI_H
